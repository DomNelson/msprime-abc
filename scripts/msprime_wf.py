import simuOpt
import attr
simuOpt.setOptions(alleleType='lineage', quiet=True)
import io
import msprime
import simuPOP as sim
import numpy as np
import sys, os
from profilehooks import timecall
import argparse


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


def evolve_lineage(N, L, n_gens, n_loci=1000, rho=1e-8):
    """
    Traces the lineage of a given number of evenly-spaced loci along a single
    chromosome, in a randomly mating population of size N
    """
    ## Initialize population, making sure that ind IDs start at 1
    sim.IdTagger().reset(1)
    pop = sim.Population(
            N,
            loci=[n_loci],
            infoFields=['ind_id', 'chromosome_id', 'allele_id', 'describe'])

    ## Create memory buffer to receive file-type output from population
    ## at each generation
    f = io.StringIO()

    ## Convert rho into an intensity along a simulated chromosome of
    ## length 'n_loci'
    intensity = L * rho / n_loci

    simu = sim.Simulator(pop, stealPops=True, rep=n_gens)
    simu.evolve(
            initOps=[
                sim.InitSex(),
                sim.IdTagger(),
                sim.InitLineage(mode=sim.FROM_INFO_SIGNED)
            ],
            matingScheme=sim.RandomMating(
                    ops=[sim.Recombinator(intensity=intensity),
                         sim.IdTagger()]),
            postOps=sim.InfoEval('ind.lineage()', exposeInd='ind', output=f),
            gen=1
        )

    lineage = f.getvalue()
    f.close()

    return lineage


def parse_lineage(lineage):
    """
    Lineage is output serially and needs to be parsed as a string. Note that
    ind IDs are 1-indexed, and positive/negative indicates maternal/paternal
    origin
    """
    ##TODO Probably more efficient strategies +t3
    lineage = lineage.strip('[]')
    lineage = lineage.split('][')

    parsed = []
    for l in lineage:
        extract_ind = lambda x: int(x.strip('[ ]'))
        parsed.append(list(map(extract_ind, l.split(','))))

    return np.asarray(parsed)

def zero_index(lineage):
    """
    Convert 1-indexed signed lineage IDs into signed 0-indexed parent IDs,
    and assign unique IDs for each generation
    """
    ## Shift abs(ID) to abs(ID)-1
    lineage = lineage - np.sign(lineage)

    return lineage.astype(int)


def generation_index(lineage):
    """
    Return array where values denote the index in the generation prior that
    the individual inherited the allele from
    """
    ##TODO Currently depends on implementation detail of replicates +t1
    n_gens, n_inds, n_loci = lineage.shape
    x = np.ones((n_inds, n_loci))
    shift_rows = np.stack([x * i * n_inds for i in range(n_gens)])

    lineage = lineage - np.sign(lineage) * shift_rows

    return lineage.astype(int)


def parent_idx(lineage):
    """
    Converts ind IDs in lineage into the indices of parents in the previous
    generation
    """
    lineage = zero_index(lineage)
    lineage = unsign_ID(lineage)
    lineage = generation_index(lineage)

    return lineage


def unsign_ID(lineages):
    """
    Takes signed maternal/paternal chromosomal IDs and converts to unsigned,
    by doubling number of IDs
    """
    ## Update indices to match, remembering that sign indicates maternal
    ## or paternal inheritance
    pos_vals = np.sign(np.abs(lineages) + lineages)
    lineages = np.abs(lineages) * 2 + pos_vals

    return lineages


def split_chroms(lineages):
    """
    Takes concatenated chromosomes and splits them into two rows
    """
    i, j, k = lineages.shape
    lineages = lineages.reshape(i, j*2, k/2)

    return lineages


def index_to_ID(idx, gen):
    """
    Converts the index of an array storing inheritance at the specified
    generation, with a row for each chromosome copy, into the signed
    ID of the same individual
    """
    ##TODO Complete +t1
    pass


def active_alleles(masked_alleles):
    """
    Returns alleles which can be traced to the sampled generation, namely
    those which have not been masked
    """
    return masked_alleles[~masked_alleles.mask].data


def find_coalescence(allele_vec):
    """
    Finds coalescent events in the current allele state
    """
    alleles_to_track = active_alleles(allele_vec)

    for allele in set(alleles_to_track):
        ##HACK - np.where evaluates masked values == 0 as True
        if allele == 0:
            inherits = np.array([i for i in range(len(allele_vec))
                                    if allele_vec[i] == 0])
        else:
            inherits = np.where(allele_vec == allele)[0]

        if len(inherits) > 1:
            yield allele, inherits


def step_lineage(allele_vec, inheritance_vec):
    """
    Returns a new allele vector updated to account for inheritances and
    coalescences
    """
    new_alleles = allele_vec[inheritance_vec]

    return new_alleles


def trace_lineage(allele_history):
    """
    Follows alleles backwards through a forward-time inheritance simulation
    """
    alleles = np.ma.masked_array(np.arange(allele_history.shape[1]))
    alleles.mask = np.zeros(alleles.shape)

    for i, inheritance_vec in enumerate(allele_history[::-1]):
        ## Follow alleles through inheritance vector, masking alleles that
        ## have coalesced
        alleles = step_lineage(alleles, inheritance_vec)

        for allele, inherits in find_coalescence(alleles):
            yield allele, inherits, i

            ## Update mask with all except first allele coalescing
            alleles.mask[inherits[1:]] = 1


@attr.s
class ForwardTrees(object):
    lineage = attr.ib()


    def __attrs_post_init__(self):
        self.parent_indices = parent_idx(self.lineage)
        self.n_gens, self.n_chroms, self.n_loci = self.lineage.shape


    def allele_history(self, allele_num):
        """
        Returns the forward-time inheritance paths for each generation, for
        the given allele
        """
        return np.ma.masked_array(self.parent_indices[:, :, allele_num])


    def ancestor(self, allele_num, gen):
        """
        Returns the ancestors carrying the allele at the given generation
        """
        return self.lineage[:, :, allele_num][gen]



def main(args):
    ## Generate tree sequence
    ts = generate_source_pops(args)

    ## Parse haplotypes
    haplotypes = msprime_hap_to_simuPOP(ts)
    positions = msprime_positions(ts)

    ## Initialize simuPOP population
    if not args.coarse:
        ## Simulate msprime haplotypes explicitly
        pop = wf_init(haplotypes, positions)
        pop = evolve_pop(pop, ngens=args.t_admix, rho=args.rho)

        ## Output genotypes
        # genotypes = [ind.genotype() for ind in pop.individuals()]
        # print(genotypes[-1])

    else:
        ## Trace lineages of discrete sections of the chromosome
        lineage = evolve_lineage(args.Na, args.length, args.t_admix, args.n_loci)
        lineage = parse_lineage(lineage)

        ## Split lineage into discrete generations
        lineage = np.array(np.split(lineage, args.t_admix))

        ## Split chromosomes into their own rows
        lineage = split_chroms(lineage)

        ## Initialize ForwardTree class
        FT = ForwardTrees(lineage)

        allele_history = FT.allele_history(0)
        labels = range(len(allele_history[0]))

        L = trace_lineage(allele_history)
        C = [l for l in L]

        print("Done")


if __name__ == "__main__":
    args = argparse.Namespace(
            Na=10,
            Ne=100,
            t_admix=3,
            t_div=10,
            admixed_prop=0.5,
            rho=1e-8,
            mu=1e-8,
            length=1e8,
            coarse=True,
            n_loci=10
            )

    main(args)
