import numpy as np
import sys, os
from profilehooks import timecall
import argparse

from wf_trace import trace
from simulate import forward_sim as fsim


def main(args):
    ## Initialize simuPOP population
    if not args.coarse:
        ## Generate tree sequence
        ts = generate_source_pops(args)

        ## Parse haplotypes
        haplotypes = msprime_hap_to_simuPOP(ts)
        positions = msprime_positions(ts)

        ## Simulate msprime haplotypes explicitly
        pop = wf_init(haplotypes, positions)
        pop = evolve_pop(pop, ngens=args.t_admix, rho=args.rho)

        ## Output genotypes
        # genotypes = [ind.genotype() for ind in pop.individuals()]
        # print(genotypes[-1])

    else:
        ## Trace lineages of discrete sections of the chromosome
        FSim = fsim.ForwardSim(args.Na,
                               args.length,
                               args.t_admix,
                               args.n_loci)
        FSim.evolve()

        ## Initialize population
        P = trace.Population(FSim)
        P.recombine()
        P.climb()
        # P.coalesce()
        # import ipdb; ipdb.set_trace()
        from IPython import embed; embed()
        # FT = trace.ForwardTrees(lineage)
        #
        # L = FT.trace_lineage(0)
        # C = [l for l in L]
        #
        # print(FT.allele_history(0))
        # print(C)


if __name__ == "__main__":
    args = argparse.Namespace(
            Na=6,
            Ne=100,
            t_admix=3,
            t_div=10,
            admixed_prop=0.5,
            rho=1e-8,
            mu=1e-8,
            length=1e8,
            coarse=True,
            n_loci=3
            )

    main(args)
