import numpy as np
import msprime
import sys, os
import argparse

import forward_sim as fsim
import pop_models


def get_genotypes(simuPOP_pop):
    """
    Returns the genotypes of each individual in the population, collected
    by subpopulation
    """
    for i in range(simuPOP_pop.numSubPop()):
        yield [ind.genotype() for ind in simuPOP_pop.individuals()] 


def hybrid_sim(ts, rho, mu, n_gens, ploidy=2, migmat=None):
    """
    Return a simuPOP population resulting from evolving the samples of the
    provided tree sequence forwards n_gens generations
    """
    ## Create a special container for a simuPOP population, which stores
    ## attributes about the initializing tree sequence
    msprime_pop = pop_models.MSPpop(ts=ts, rho=rho, mu=mu, ploidy=ploidy,
                                    migmat=migmat)
    simuPOP_pop = pop_models.msp_to_simuPOP(msprime_pop)

    ## Perform forward simulations
    FSim = fsim.ForwardSim(n_gens, simuPOP_pop)
    FSim.evolve()

    return FSim.pop


def main(args):
    ts = msprime.simulate(args.n_inds * args.ploidy,
            recombination_rate=args.rho, mutation_rate=args.mu,
            length=args.L, Ne=args.Ne)

    H = hybrid_sim(
            ts=ts,
            rho=args.rho,
            mu=args.mu,
            n_gens=args.n_gens,
            ploidy=args.ploidy)

    return H, ts


if __name__ == "__main__":
    args = argparse.Namespace(
            n_inds=200,
            Ne=100,
            n_gens=10,
            rho=1e-8,
            mu=1e-10,
            L=1e8,
            ploidy=2,
            )

    H, ts = main(args)
