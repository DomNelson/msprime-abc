import simuOpt
simuOpt.setOptions(alleleType='lineage', quiet=True)
import attr
import io
import msprime
import simuPOP as sim
import numpy as np
import sys, os


def generate_source_pops(args):
    ## Initialize admixed and source populations with two chromosomes per ind
    pop0_size = int(2 * args.Na * args.admixed_prop)
    pop1_size = 2 * args.Na - pop0_size

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
                    time=args.t_div,
                    source=0,
                    destination=1,
                    proportion=1.)]
            
    ## Coalescent simulation
    ts = msprime.simulate(
            population_configurations=population_configurations,
            demographic_events=demographic_events,
            recombination_rate=args.rho,
            length=args.length,
            mutation_rate=args.mu,
            Ne=args.Ne)

    return ts


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
    pop = sim.Population(size=[len(haplotypes)/2], ploidy=ploidy)
    pop.addChrom(positions, chromName='0')
    
    ## Set genotypes for each individual separately
    ##TODO Probably a more efficient way of setting genotypes +t2
    for ind, gen in zip(pop.individuals(), haplotypes):
        ind.setGenotype(gen)

    return pop


def evolve_pop(pop, ngens, rho, rep=1, mutation_matrix=None):
    ## Initialize simulator, without modifying original population
    simu = sim.Simulator(pop, stealPops=False, rep=rep)

    if mutation_matrix is None:
        ##TODO Confirm only ref/alt in msprime mutation model (inf sites) +t2
        mutation_matrix = np.identity(2).tolist()

    ## Evolve as randomly mating population without new mutations
    simu.evolve(
            initOps=sim.InitSex(),
            preOps=sim.MatrixMutator(rate=mutation_matrix),
            matingScheme=sim.RandomMating(
                    ops=sim.Recombinator(intensity=rho)),
            gen=ngens)

    ##TODO This only returns one replicate even if more were specified +t1
    newpop = simu.extract(0)

    return newpop


@attr.s
class ForwardSim(object):
    N = attr.ib()
    L = attr.ib()
    n_gens = attr.ib()
    n_loci = attr.ib(default=1000)
    rho = attr.ib(default=1e-8)
    output = attr.ib(default='genotypes.txt')


    def evolve(self):
        """
        Traces the lineage of a given number of evenly-spaced loci along a
        single chromosome, in a randomly mating population of size N
        """
        ## Initialize population, making sure that ind IDs start at 1
        sim.IdTagger().reset(1)
        pop = sim.Population(
               self.N,
               loci=[self.n_loci],
               infoFields=['ind_id', 'chromosome_id', 'allele_id', 'describe'])

        ## Create memory buffer to receive file-type output from population
        ## at each generation
        ID = io.StringIO()
        recs = io.StringIO()

        get_ID = 'str(int(ind.info("ind_id"))) + "\t"'
        get_genotype = "str(int(ind.info('ind_id'))) + ','" +\
                        "+ str(ind.genotype()).strip('[]') + '\\n'"
        get_genotype_init = "str(-1 * int(ind.info('ind_id'))) + ','" +\
                        "+ str(ind.genotype()).strip('[]') + '\\n'"

        ## Convert rho into an intensity along a simulated chromosome of
        ## length 'n_loci'
        intensity = self.L * self.rho / self.n_loci

        simu = sim.Simulator(pop, stealPops=True, rep=1)
        simu.evolve(
            initOps=[sim.IdTagger(),
                sim.InitSex(),
                sim.InitGenotype(freq=[0.2, 0.8]),
                sim.InitLineage(mode=sim.FROM_INFO_SIGNED),
                sim.InfoEval(get_ID, exposeInd='ind', output=ID),
                sim.InfoEval(get_genotype_init, exposeInd='ind', output=self.output)
            ],
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

        recs.close()
        ID.close()

        ## Set parsed attributes
        self.parse_sim()
        self.pop = simu.extract(0)


    def parse_sim(self):
        ## Represent chromosomes as signed individual IDs
        signed_ID = np.stack([self.raw_ID, self.raw_ID * -1]).T
        self.ID = parse_output(signed_ID, self.n_gens)

        ## Recombinations stay as a list of lists
        self.recs = sort_recombs(self.raw_recs)


    def get_idx(self, ID):
        idx = 2 * np.abs(ID) - (np.sign(ID) - 1) / 2 - 2

        ## Zero ID will give a negative number, which indicates no ind
        idx[idx < 0] = -1

        return idx.astype(int)


    def ind_haplotypes(self):
        """
        Returns an array of genotypes and corresponding chromosome IDs
        """
        data = np.genfromtxt(self.output, delimiter=',', dtype=int)

        ## Split chromosomes into separate rows
        haplotypes = data[:, 1:].reshape(data.shape[0] * 2, data.shape[1] / 2)

        ## Signed labels for chroms, where first chrom has negative ind ID
        ##TODO: Verify this is consistent with simuPOP labels +t1
        ind_IDs = data[:, 0]
        chrom_IDs = np.empty((ind_IDs.size * 2,), dtype=ind_IDs.dtype)
        chrom_IDs[0::2] = ind_IDs * -1
        chrom_IDs[1::2] = ind_IDs

        return chrom_IDs, haplotypes


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
    
    return lineages.reshape(i*2, j/2)


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





