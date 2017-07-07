import numpy as np
import sys, os
from profilehooks import timecall
import argparse
import configparser

import forward_sim as fsim
import wf_trace as trace


def main(args):
    ## Initialize simuPOP population
    if args.forward:
        ## Generate tree sequence
        ts = fsim.generate_source_pops(args)

        ## Parse haplotypes
        haplotypes = fsim.msprime_hap_to_simuPOP(ts)
        positions = fsim.msprime_positions(ts)

        ## Simulate msprime haplotypes explicitly
        pop = fsim.wf_init(haplotypes, positions)
        pop = fsim.evolve_pop(pop, ngens=args.t_admix, rho=args.rho)

        ## Output genotypes
        genotypes = [ind.genotype() for ind in pop.individuals()]
        # print(genotypes[-1])

    else:
        ## Trace coalescent trees through lineages generated in forward manner
        FSim = fsim.ForwardSim(args.Na,
                               args.length,
                               args.t_admix,
                               args.n_loci,
                               args.rho)
        FSim.evolve()
        FSim.write_haplotypes('gen.h5')

        ## Initialize population and trace haplotype lineages
        ID = FSim.ID.ravel()
        recs = FSim.recs
        P = trace.Population(ID, recs, args.t_admix, args.n_loci)
        P.trace()

        ## Convert traces haplotypes into an msprime TreeSequence
        positions = FSim.pop.lociPos()
        T = trace.WFTree(P.haps, positions)
        ts = T.tree_sequence()
        ts.dump('treesequence.h5')

        ## Check node genotypes
        for t in ts.trees():
            left, right = list(map(int, t.interval))
            print("interval:", left, right)

            genotypes = T.node_genotypes(t.nodes(), 'gen.h5')

            for r in t.children(t.root):
                print("Root node", r)
                print("genotypes:")
                for l in t.leaves(r):
                    print(genotypes[l][left:right])


        ## Collect sparse trees and plot a selection
        # trees = list(ts.trees())
        # for i, t in enumerate(trees[:5]):
        #     t.draw('tree_' + str(i) + '.svg', width=5000, height=500,
        #             show_times=True)

        return P, T, ts, t, left, right, genotypes


if __name__ == "__main__":
    args = argparse.Namespace(
            Na=10,
            Ne=100,
            t_admix=2,
            t_div=100,
            admixed_prop=0.5,
            rho=1e-8,
            mu=1e-8,
            length=1e8,
            forward=False,
            n_loci=20
            )

    # P = main(args)
    P, T, ts, t, left, right, genotypes = main(args)
