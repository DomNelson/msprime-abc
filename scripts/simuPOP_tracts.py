import sys, os
import numpy as np
import attr
import copy
import argparse
from profilehooks import profile, timecall
from collections import defaultdict, Counter
import simuPOP as sim


def tract_lengths(haplotype):
    """ Returns tract lengths of each ancestry """
    ##TODO Make sure to split chomosome copies +t1
    haplotype = np.array(haplotype)
    breakpoints = np.where(np.diff(haplotype) != 0)[0]
    tracts = defaultdict(list)
    binned_tracts = defaultdict(Counter)

    ## If there are no breakpoints, return a single tract
    if len(breakpoints) == 0:
        ancestry = haplotype[0]
        binned_tracts[ancestry] = Counter([len(haplotype)])

        return binned_tracts

    ## Get the length of the first and last tracts
    tracts[haplotype[0]].append(breakpoints[0] + 1)
    tracts[haplotype[-1]].append(len(haplotype)-breakpoints[-1] - 1)

    ## Tract lengths are the difference between successive breakpoints
    lengths = np.ediff1d(breakpoints)
    for l, b in zip(lengths, breakpoints[1:]):
        tracts[haplotype[b]].append(l)

    ## Bin tracts by length for each ancestry
    for k, v in tracts.items():
        binned_tracts[k] = Counter(v)

    return binned_tracts


def ancestor_counts(n_inds, ancestry_props):
    """ Returns number of inds with each ancestry, in list format """
    inds_per_ancestry = (np.array(ancestry_props) * n_inds).astype(int)
    return list(inds_per_ancestry)


def pop_haplotypes(pop):
    ##TODO Chroms of same individual always adjascent? Write test +t1
    genotypes = np.asarray(pop.genotype())

    ##TODO Only works for single chromosome +t2 +n1
    n_sites = pop.numLoci()[0]
    haplotypes = genotypes.reshape(-1, n_sites)

    return haplotypes


def initialize_pop(n_inds, n_sites, ancestry_props):
    """
    Initializes a simuPOP population using the provided parameters
    """
    inds_per_anc = ancestor_counts(n_inds, ancestry_props)
    pop = sim.Population(size=inds_per_anc, ploidy=2, loci=n_sites)

    ## Set population attributes
    pop.dvars().n_ancestries = len(ancestry_props)

    ## Set genotypes for each source population as the index of their ancestry
    for i in range(len(ancestry_props)):
        sim.initGenotype(pop, haplotypes=[i]*n_sites, subPops=[i])

    ## Merge subpopulations
    pop.mergeSubPops(range(pop.dvars().n_ancestries))

    return pop


def evolve_pop(pop, ngens, rep=1, mutation_matrix=None):
    ## Initialize simulator, without modifying original population
    simu = sim.Simulator(pop, stealPops=False, rep=rep)

    if mutation_matrix is None:
        mutation_matrix = np.identity(pop.dvars().n_ancestries).tolist()

    ## Evolve with given parameters
    simu.evolve(
        initOps=sim.InitSex(),
        preOps=sim.MatrixMutator(rate=mutation_matrix),
        matingScheme=sim.RandomMating(
                ops=sim.Recombinator(intensity=0.1)),
        gen=ngens
    )

    ##TODO This only returns one replicate +t1
    newpop = simu.extract(0)

    return newpop


def evolve_genotypes(pop, ngens):
    """ Evolves the population ngens, and returns genotypes """
    ##TODO Bug?? Using this instead of calling pop_genotypes outside +t2
    ##TODO of class gives different results, namely with mutations +t2
    print("Deprecated - use at own risk")
    pop = evolve_pop(pop, ngens)

    return pop_genotypes(pop)


def pop_tracts(pop):
    """ Returns binned tract lengths for the provided population """
    haplotypes = pop_haplotypes(pop)

    ##TODO Change to pass haplotypes +t1
    all_tracts = copy.copy(tract_lengths(haplotypes[0]))

    for g in haplotypes[1:]:
        for k, v in tract_lengths(g).items():
            all_tracts[k].update(v)

    return all_tracts


@timecall
def main(args):
    p = initialize_pop(args.N, args.n_sites, args.props)
    p_2 = evolve_pop(p, args.n_gens)
    # g_2 = pop_haplotypes(p_2)
    # t_2 = tract_lengths(g_2[-1])
    # print(g_2[-1])
    # print(t_2)
    # print(pop_tracts(p_2))


if __name__ == "__main__":
    args = argparse.Namespace(
            N=1000,
            n_gens=100,
            n_sites=5000,
            props=[0.3, 0.2, 0.4, 0.1],
            )

    main(args)
