import simuOpt
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


def parent_idx(lineage, n_gens):
    """
    Convert 1-indexed signed lineage IDs into signed 0-indexed parent IDs,
    and assign unique IDs for each generation
    """
    ##TODO Currently depends on implementation detail of replicates +t1
    n_inds = lineage.shape[0] / n_gens
    x = np.ones((n_inds, 1))
    shift_rows = np.vstack([x * i * n_inds for i in range(n_gens)])

    ## Shift abs(ID) to abs(ID)-1
    lineage = lineage - np.sign(lineage)

    ## Make all IDs < n_inds
    lineage = lineage - np.sign(lineage) * shift_rows

    ## Split into one array per generation
    lineage = np.array(np.split(lineage, n_gens))

    return lineage.astype(int)


def split_chroms(lineages):
    """
    Takes concatenated chromosomes and splits them into two rows
    """
    i, j, k = lineages.shape
    lineages = lineages.reshape(i, j*2, k/2)

    ## Update indices to match, remembering that sign indicates maternal
    ## or paternal inheritance
    pos_vals = np.sign(np.abs(lineages) + lineages)
    lineages = np.abs(lineages) * 2 + pos_vals

    return lineages


def find_coalescence(allele_lineage, ancs, gen):
    """ Finds coalescences among active lineages 'ancs' """
    coals = []
    for anc in ancs:
        inherits = np.where(allele_lineage == anc)

        ## More than one inheritance of an allele means a coalescence
        if len(inherits[0]) > 1:
            coals.append((inherits[0], anc, gen))

    return coals


def trace_allele_lineage(allele_lineage):
    """ Reconstructs coalescent tree from wf simulations """
    ## Recursively trace alleles back through the generations
    current_gen = allele_lineage[-1]
    ancs = set(current_gen)

    coals = [find_coalescence(current_gen, ancs, len(allele_lineage)-1)]
    for gen in range(len(allele_lineage)-1, 0, -1):
        current_gen = allele_lineage[gen-1][current_gen]

        ## Tract coalescence events
        ##TODO Think about how to vectorize +t3
        ancs = set(allele_lineage[gen-1]).intersection(ancs)
        coals.append(find_coalescence(allele_lineage[gen-1], ancs, gen-1))

    return current_gen, coals


def trace_lineage(lineage):
    n_gens = lineage.shape[0]

    all_trace = []
    all_coals = []
    for i in range(lineage.shape[2]):
        allele_lineage, coals = trace_allele_lineage(lineage[:, :, i])
        all_trace.append(allele_lineage)
        all_coals.append(coals)

    
    return np.asarray(all_trace).T, all_coals


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

    else:
        ## Trace lineages of discrete sections of the chromosome
        lineage = evolve_lineage(args.Na, args.length, args.t_admix, args.n_loci)
        lineage = parse_lineage(lineage)
        parent_indices = parent_idx(lineage, args.t_admix)
        print(parent_indices.shape)

        parent_indices = split_chroms(parent_indices)
        print(parent_indices)
        print(parent_indices.shape)

        allele_origins, coals = trace_lineage(parent_indices)
        print(allele_origins)
        print(coals)

    ## Output genotypes
    # genotypes = [ind.genotype() for ind in pop.individuals()]
    # print(genotypes[-1])


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
