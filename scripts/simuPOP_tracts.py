import sys, os
import numpy as np
import simuPOP as sim


def pop_init(n_inds, n_sites, ancestry_props):
    """
    Initializes a simuPOP population using the provided haplotypes
    """
    inds_per_ancestry = list((np.array(ancestry_props) * n_inds).astype(int))
    pop = sim.Population(size=inds_per_ancestry, ploidy=2, loci=n_sites)
    
    ## Set genotypes for each source population as the index of their ancestry
    for i in range(len(ancestry_props)):
        sim.initGenotype(pop, haplotypes=[i]*n_sites, subPops=[i])

    ## Merge subpopulations
    pop.mergeSubPops(range(len(ancestry_props)))

    return pop


def evolve_pop(pop, n_ancestries, ngens):
    ## Initialize simulator
    simu = sim.Simulator(pop, stealPops=False)

    ## Evolve with given parameters
    simu.evolve(
        initOps=sim.InitSex(),
        preOps=sim.MatrixMutator(rate = np.identity(n_ancestries).tolist()),
        matingScheme=sim.RandomMating(),
        finalOps=sim.SavePopulation('!"pop%d.pop"%rep'),
        gen=ngens
    )

    newpop = simu.extract(0)

    return newpop

