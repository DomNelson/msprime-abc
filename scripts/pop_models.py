import numpy as np
import attr
import msprime
import simuOpt
simuOpt.setOptions(alleleType='lineage', quiet=True)
import simuPOP as sim
from collections import defaultdict, Counter


@attr.s
class SimuPOPpop(object):
    ## Population parameters
    rho = attr.ib()
    mu = attr.ib()

    ## simuPOP population
    pop = attr.ib()

    migmat = attr.ib(default=None)


@attr.s
class MSPpop(object):
    ## Population parameters
    rho = attr.ib()
    mu = attr.ib()

    ## msprime tree sequence
    ts = attr.ib()

    ploidy = attr.ib(default=2)
    migmat = attr.ib(default=None)


def msprime_hap_to_simuPOP(ts, ploidy=2):
    """
    Takes msprime haplotypes and returns them in a format readable by simuPOP,
    sorting by population by default
    """
    ## Extract haplotypes in simuPOP format
    haplotypes = ts.haplotypes()
    simuPOP_haps = [list(map(int, list(str(x)))) for x in haplotypes]

    ## Get population of samples
    N = ts.sample_size
    pops = [n.population for i, n in enumerate(ts.nodes()) if i < N]

    ## Sort haplotypes by population
    dtypes = [('pops', int), ('haps', list)]
    tuples = list(zip(pops, simuPOP_haps))
    hap_tuples = np.core.records.fromrecords(tuples, dtype=dtypes)
    sorted_tuples = np.sort(hap_tuples, order=['pops'])

    ## Truncate rows which can't be used to create individuals
    extra_rows = sorted_tuples.shape[0] % ploidy
    if extra_rows != 0:
        sorted_tuples = sorted_tuples[:-extra_rows]

    ## Concatenate neighbouring haplotypes to form inds, most of which will
    ## have both chromosomes from the same population
    haps = np.vstack(sorted_tuples['haps'])
    inds = haps.reshape(int(haps.shape[0] / ploidy), -1)

    ## Ind pop is the pop of their first chromosome
    pops = sorted_tuples['pops'][::ploidy]

    return inds.tolist(), pops.astype(float)


def msprime_pop_freqs(ts):
    """
    Returns the allele frequency of each locus in a tree sequence, collected
    by population
    """
    ## Set ploidy to 1 so we pull out individual haplotypes, and loci will
    ## align properly
    inds, pops = msprime_hap_to_simuPOP(ts, ploidy=1)

    ## Collect inds by population
    pop_inds = defaultdict(list)
    for ind, pop in zip(inds, pops):
        pop_inds[pop].append(ind)

    ## Calculate allele frequencies for each pop
    for pop, ind_list in pop_inds.items():
        allele_counts = np.vstack(ind_list).sum(axis=0)
        
        yield pop, allele_counts / len(ind_list)


def simuPOP_pop_freqs(pop, loci=True, ploidy=2):
    """
    Returns the allele frequency of specified loci in each subpopulation of a
    simuPOP population, defaulting to all loci
    """
    for i in range(pop.numSubPop()):
        inds = np.vstack([ind.genotype() for ind in pop.individuals(i)])
        inds = inds.reshape(inds.shape[0]*ploidy, int(inds.shape[1]/ploidy))

        if loci is True:
            allele_counts = inds.sum(axis=0)
        else:
            allele_counts = inds[loci].sum(axis=0)

        yield i, allele_counts / len(inds)


def msprime_positions(TreeSequence):
    """ Returns position of mutations in TreeSequence """
    ##TODO Could be done in the haplotype loop above +t3
    return [site.position for site in TreeSequence.sites()]


def msp_to_simuPOP(msp_pop):
    """
    Returns a simuPOP population initialized with the results of the
    msprime coalescent simulation
    """
    haps, pops = msprime_hap_to_simuPOP(msp_pop.ts)
    positions = msprime_positions(msp_pop.ts)

    simuPOP_pop = wf_init(haps, positions, pops, msp_pop.ploidy)

    return SimuPOPpop(rho=msp_pop.rho, mu=msp_pop.mu, pop=simuPOP_pop,
                        migmat=msp_pop.migmat)


def wf_init(haplotypes, positions, populations=None, ploidy=2):
    """
    Initializes a simuPOP population using the provided haplotypes
    """
    ## If no populations are specified, all inds come from pop 0
    if populations is None:
        populations = np.zeros(len(haplotypes)) - 1

    ## Make sure populations are sorted
    assert np.min(np.diff(populations)) >= 0

    ## Get size of each population
    sizes = pop_sizes(populations)

    ##NOTE: Assumes a single chromosome +n1
    ##TODO: Should track haplotypes so we can connect back after sims +t2
    pop = sim.Population(size=sizes, ploidy=ploidy)
    pop.addLoci([0] * len(positions), positions)

    ## Set attributes we need to track lineages
    info_fields = ['ind_id', 'chromosome_id', 'allele_id', 'describe',
                    'migrate_to']
    pop.setInfoFields(info_fields)
    
    ## Set genotypes for each individual separately
    for ind, gen, pop_label in zip(pop.individuals(), haplotypes, populations):
        ind.setGenotype(gen)
        ind.migrate_to = pop_label

    return pop


def pop_sizes(pops):
    """
    Returns a sorted list of sizes of each population
    """
    counts = Counter(pops)

    sizes = []
    for pop in sorted(set(pops)):
        sizes.append(counts[pop])

    return sizes


def maf_init_simuPOP(N, rho, L, mu, MAF, ploidy=2, migmat=None):
    """
    Returns a simuPOP population for forward simulations by specifying
    a MAF in the founding generation.
    
    Allele frequencies (MAF) are specified as an array, with a row for each
    subpopulation and a column for each locus. MAF can also be provided as a
    float, if a single population with a single is desired

    N should be a vector of sub-population sizes, unless MAF is a float or has
    a single row, in which case it can be an integer
    """
    if type(MAF) is float:
        MAF = np.array(MAF).reshape(1, 1)
    else:
        assert len(MAF.shape) == 2

    if type(N) is int:
        N = np.array(N).reshape(1,)

    ## Make sure proper number of populations have been specified
    n_pops, n_loci = MAF.shape
    assert N.shape[0] == n_pops
    if n_pops > 1:
        assert len(migmat) == len(migmat[0]) == n_pops, \
                "Must specify migration matrix for multiple populations"

    ## Draw random loci along the genome 
    positions = np.random.uniform(0, L, size=n_loci)

    ## Draw random genotypes and flatten pops together
    pops, genotypes = zip(*list(draw_pop_genotypes(N, MAF, ploidy)))

    ## Convert to list with type expected by simuPOP
    positions = list(map(float, positions))
    genotypes = [list(map(int, g)) for row in genotypes for g in row]

    simuPOP_pop = wf_init(genotypes, positions, pops, ploidy)

    return SimuPOPpop(rho=rho, mu=mu, pop=simuPOP_pop, migmat=migmat)


def draw_genotypes(MAF_array):
    """
    Draws genotypes given an array specifying allele frequencies by
    population (row) and locus (column), and outputs as a list of lists
    for initializing a simuPOP population
    """
    n_pops, n_loci = MAF_array.shape
    draw = np.random.random(size=(n_pops, n_loci))

    return (draw < MAF_array).astype(int).tolist()


def draw_pop_genotypes(N_array, MAF_array, ploidy=2):
    """
    Draws genotypes for sub-population sizes specified in N_array, for
    frequencies specified by population (row) and locus (column) in MAF_array
    """
    for pop, n in enumerate(N_array):
        for i in range(n):

            yield pop, draw_genotypes(MAF_array)


def simple_pop_ts(N, rho, L, mu, Ne, ploidy=2):
    """
    Returns an MSPpop instance created from a single randomly mating population
    """
    ts = msprime.simulate(sample_size=N*ploidy, Ne=Ne, length=L,
            recombination_rate=rho, mutation_rate=mu)

    return MSPpop(rho, mu, ts, ploidy)


def diverged_pops_ts(N, rho, L, mu, Ne, t_div, admixed_prop, ploidy=2):
    """
    Generates a tree sequence associated with two diverged populations and
    returns an MSPpop instance
    """
    ## Initialize admixed and source populations with two chromosomes
    ## per ind
    pop0_size = int(2 * N * admixed_prop)
    pop1_size = 2 * N - pop0_size

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
                    time=t_div,
                    source=0,
                    destination=1,
                    proportion=1.)]
            
    ## Coalescent simulation
    ts = msprime.simulate(
            population_configurations=population_configurations,
            demographic_events=demographic_events,
            recombination_rate=rho,
            length=L,
            mutation_rate=mu,
            Ne=Ne)

    return MSPpop(rho, mu, ts)


def grid_ts(N, rho, L, mu, Ne, t_div, mig_prob, grid_width, ploidy=2):
    """
    Returns a tree sequence associated with a population dispersed over a
    grid of randomly-mating demes, with nearest-neighbour migration
    """
    ## Initialize the population within each deme
    ##TODO: Set separate Ne? +t2
    pop_conf = []
    N_deme = np.round(N / grid_width**2).astype(int)
    for i in range(grid_width**2):
        pop_conf.append(msprime.PopulationConfiguration(sample_size=N_deme))

    ## Set initial source population, which is simply one of the demes - note
    ## that all demes are still initialized equally
    dem_events = [msprime.MigrationRateChange(time=t_div, rate=0)]
    for i in range(1, grid_width**2):
            dem_events.append(msprime.MigrationRateChange(time=t_div, rate=1,
                    matrix_index=(i, 0)))

    migmat = grid_migration(grid_width, mig_prob)

    ts = msprime.simulate(population_configurations=pop_conf,
                migration_matrix=migmat, recombination_rate=rho,
                demographic_events=dem_events, mutation_rate=mu,
                length=L, Ne=Ne)

    return MSPpop(rho, mu, ts, ploidy, migmat)


def grid_migration(n, mig_prob):
    """
    Returns a migration matrix for a grid of demes with nearest-neighbour
    migrations. Diagonal elements must be zero.
    """
    ##TODO: Ouch... can we make a sparse representation? +t2
    migmat = np.zeros((n**2, n**2))
    
    for i in range(n**2):
        ## Add next node if we're not at the right edge
        if i % n != n-1:
            migmat[i, i+1] = mig_prob
            
        ## Add node below if we're not at the lower edge
        if n**2 - i > n:
            migmat[i, i+n] = mig_prob
            
    ## Convert upper triangular matrix to symmetric matrix
    migmat = migmat + migmat.T
    
    return migmat.tolist()
