import simuOpt
import tables
simuOpt.setOptions(alleleType='lineage', quiet=True)
import attr
import io
import msprime
import simuPOP as sim
import numpy as np
import sys, os
from collections import namedtuple


@attr.s
class Pop(object):
    ## Population parameters
    rho = attr.ib()
    mu = attr.ib()

    ## simuPOP population
    pop = attr.ib()


@attr.s
class MsprimePop(object):
    N = attr.ib()
    rho = attr.ib()
    L = attr.ib()
    mu = attr.ib()
    Ne = attr.ib()
    t_div = attr.ib()
    admixed_prop = attr.ib()
    ploidy = attr.ib(default=2)


    def __attrs_post_init__(self):
        self.ts = self.generate_source_pops()


    def generate_source_pops(self):
        ## Initialize admixed and source populations with two chromosomes
        ## per ind
        pop0_size = int(2 * self.N * self.admixed_prop)
        pop1_size = 2 * self.N - pop0_size

        population_configurations = [
                msprime.PopulationConfiguration(
                        sample_size=pop0_size,
                        growth_rate=0),
                msprime.PopulationConfiguration(
                        sample_size=pop1_size,
                        growth_rate=0)]

        ## Specify admixture event
        demographic_events = [
                msprime.MassMigration(
                        time=self.t_div,
                        source=0,
                        destination=1,
                        proportion=1.)]
                
        ## Coalescent simulation
        ts = msprime.simulate(
                population_configurations=population_configurations,
                demographic_events=demographic_events,
                recombination_rate=self.rho,
                length=self.L,
                mutation_rate=self.mu,
                Ne=self.Ne)

        return ts


    def as_simuPOP(self):
        """
        Returns a simuPOP population initialized with the results of the
        msprime coalescent simulation
        """
        haps = msprime_hap_to_simuPOP(self.ts)
        positions = msprime_positions(self.ts)

        simuPOP_pop = wf_init(haps, positions, self.ploidy)

        return Pop(rho=self.rho, mu=self.mu, pop=simuPOP_pop)


@attr.s
class MAFPop(object):
    """
    Initializes a simuPOP population for forward simulations by specifying
    a MAF in the founding generation
    """
    N = attr.ib()
    rho = attr.ib()
    L = attr.ib()
    mu = attr.ib()
    n_loci = attr.ib()
    MAF = attr.ib(convert=float)
    ploidy = attr.ib(default=2)


    def __attrs_post_init__(self):
        ##TODO: Specify allele frequencies for all loci +t1
        self.freqs = [1-self.MAF, self.MAF]
        self.pop = self.as_simuPOP()


    def as_simuPOP(self):
        """
        Draws random loci along the genome and initializes genotypes in the
        specified proportions
        """
        positions = np.random.uniform(0, self.L, size=self.n_loci)
        haps_shape = (self.N * self.ploidy, self.n_loci)
        haps = np.random.choice([0, 1], p=self.freqs, size=haps_shape)

        ## Convert to lists with types expected by simuPOP
        positions = list(map(float, positions))
        haps = [list(map(int, h)) for h in haps]

        simuPOP_pop = wf_init(haps, positions, self.ploidy)

        return Pop(rho=self.rho, mu=self.mu, pop=simuPOP_pop)


@attr.s
class ForwardSim(object):
    n_gens = attr.ib()
    initial_pop = attr.ib()
    output = attr.ib(default='genotypes.txt')


    def __attrs_post_init__(self):
        self.rho = self.initial_pop.rho
        self.mu = self.initial_pop.mu
        self.pop = self.initial_pop.pop

        ## Set attributes we need to track lineages
        info_fields = ['ind_id', 'chromosome_id', 'allele_id', 'describe']
        self.pop.setInfoFields(info_fields)

        ##NOTE: Assumes only one chromosome +n1
        assert len(self.pop.numLoci()) == 1
        self.n_loci = self.pop.numLoci()[0]
        self.L = self.pop.genoSize() / self.pop.ploidy()


    def evolve(self):
        """
        Traces the lineage of a given number of evenly-spaced loci along a
        single chromosome, in a randomly mating population of size N
        """
        ## Initialize population, making sure that ind IDs start at 1
        sim.IdTagger().reset(1)

        ## Create memory buffer to receive file-type output from population
        ## at each generation
        ID = io.StringIO()
        recs = io.StringIO()
        muts = io.StringIO()

        get_ID = 'str(int(ind.info("ind_id"))) + "\t"'
        get_genotype = "str(int(ind.info('ind_id'))) + ','" +\
                        "+ str(ind.genotype()).strip('[]') + '\\n'"
        get_genotype_init = "str(-1 * int(ind.info('ind_id'))) + ','" +\
                        "+ str(ind.genotype()).strip('[]') + '\\n'"

        ## Convert rho into an intensity along a simulated chromosome of
        ## length 'n_loci'
        intensity = self.L * self.rho / self.n_loci

        simu = sim.Simulator(self.pop, stealPops=True, rep=1)
        simu.evolve(
            initOps=[sim.IdTagger(),
                sim.InitSex(),
                sim.InitLineage(mode=sim.FROM_INFO_SIGNED),
                sim.InfoEval(get_ID, exposeInd='ind', output=ID),
                sim.InfoEval(get_genotype_init, exposeInd='ind',
                             output=self.output)
            ],
            preOps=[sim.SNPMutator(u=self.mu, output=muts)],
            matingScheme=sim.RandomMating(
                ops=[sim.IdTagger(),
                     sim.Recombinator(intensity=intensity,
                                      output=recs,
                                      infoFields='ind_id')]
                     ),
            postOps=[sim.InfoEval(get_ID, exposeInd='ind', output=ID),
                 sim.InfoEval(get_genotype, exposeInd='ind',
                              output='>>' + self.output),
                 ],
                gen=self.n_gens
            )

        self.raw_ID = list_string_to_np(ID.getvalue(), depth=1)
        self.raw_recs = recs.getvalue()
        self.muts = store_mutants(muts.getvalue())

        recs.close()
        ID.close()
        muts.close()

        ## Set parsed attributes
        self.parse_sim()
        self.pop = simu.extract(0)


    def parse_sim(self):
        ## Represent chromosomes as signed individual IDs
        signed_ID = np.stack([self.raw_ID, self.raw_ID * -1]).T
        self.ID = parse_output(signed_ID, self.n_gens)

        ## Recombinations stay as a list of lists
        self.recs = sort_recombs(self.raw_recs)


    def write_haplotypes(self, h5file):
        """
        Returns an array of genotypes and corresponding chromosome IDs
        """
        filters = tables.Filters(complevel=5, complib='blosc')

        with tables.open_file(h5file, 'w') as h5:
            with open(self.output, 'r') as f:
                line = next(f)
                ind_ID, haplotypes = parse_simuPOP_genotype(line)

		## Create an extendable array in the h5 output file with
		## the same shape as the haplotypes
                h5.create_earray(h5.root, 'haps',
                         atom=tables.IntAtom(shape=(2, haplotypes.shape[2])),
                         shape=(0,), filters=filters)
                h5.create_earray(h5.root, 'inds', atom=tables.IntAtom(),
                         shape=(0,), filters=filters)

                h5.root.haps.append(haplotypes)
                h5.root.inds.append(ind_ID)

                for line in f:
                    ind_ID, haplotypes = parse_simuPOP_genotype(line)
                    h5.root.haps.append(haplotypes)
                    h5.root.inds.append(ind_ID)


def parse_simuPOP_genotype(genotype_string):
    """
    Returns chom_IDs and haplotypes from simuPOP string output
    """
    data = np.fromstring(genotype_string, sep=',')
    ind_ID = data[0].reshape(1)
    haplotypes = data[1:].reshape(1, 2, -1)

    return ind_ID, haplotypes
        
        
def parse_output(output, n_gens, split=False):
    """
    Takes a string representation of simulation output and returns a
    numpy array with individuals having one row per chromosome, and
    the first axis representing time in generations
    """
    output = split_chroms(output)

    if split is True:
        output = split_gens(output, n_gens)

    return output


def list_string_to_np(list_str, depth, astype=int):
    """
    Converts a string representation of a list of lists into a numpy array
    """
    assert depth in [1,2]
    a = list_str.strip().split('\t')

    if depth == 2:
        a = map(lambda x: x.strip('[]').split(','), a)
                
    return np.array(list(a)).astype(astype)


def split_chroms(lineages):
    """
    Takes concatenated chromosomes and splits them into two rows
    """
    i, j = lineages.shape
    assert j % 2 == 0
    
    return lineages.reshape(i*2, int(j/2))


def split_gens(array, n_gens):
    """
    Split a single array output along a new axis representing distinct
    generation
    """
    return np.array(np.split(array, n_gens))


def sort_recombs(rec_str):
    """
    Returns a list of recombinations in the population history, in the format:
    """
    recs = rec_str.strip().split('\n')

    return list(map(lambda x: list(map(int, x.split(' '))), recs))


def msprime_hap_to_simuPOP(TreeSequence):
    """
    Takes msprime haplotypes and returns them in a format readable by simuPOP
    """
    haplotypes = TreeSequence.haplotypes()
    simuPOP_haps = [list(map(int, list(str(x)))) for x in haplotypes]

    return simuPOP_haps


def msprime_positions(TreeSequence):
    """ Returns position of mutations in TreeSequence """
    ##TODO Could be done in the haplotype loop above +t3
    return [site.position for site in TreeSequence.sites()]


def wf_init(haplotypes, positions, ploidy=2):
    """
    Initializes a simuPOP population using the provided haplotypes
    """
    ##NOTE: Assumes a single chromosome +n1
    pop = sim.Population(size=[len(haplotypes)/2], ploidy=ploidy)
    pop.addLoci([0] * len(positions), positions)
    
    ## Set genotypes for each individual separately
    ##TODO Probably a more efficient way of setting genotypes +t2
    for ind, gen in zip(pop.individuals(), haplotypes):
        ind.setGenotype(gen)

    return pop


def parse_mutants(mutant_data):
    """
    Returns each mutation event as a tuple, read form simuPOP output string
    """
    for line in mutant_data.split('\n'):
        # a trailing \n will lead to an empty string
        if not line:
            continue
        (gen, loc, ploidy, a1, a2, ID) = line.split('\t')
        yield gen, loc, ploidy, a1, a2, ID


def store_mutants(mutant_data):
    """
    Returns simuPOP mutation output as a structured numpy array
    """
    dtypes = [('gen', int), ('loc', int), ('chrom', int), ('a1', int),
              ('a2', int), ('ID', int)]
    muts = list(parse_mutants(mutant_data))

    if len(muts) > 0:
        mut_array = np.core.records.fromrecords(muts, dtype=dtypes)
    else:
        mut_array = np.array([], dtype=dtypes)

    return mut_array
