import numpy as np
import sys, os
from profilehooks import timecall
import argparse
import configparser
import attr

import forward_sim as fsim
import wf_trace as trace


@attr.s
class WFTree(object):
    n_inds = attr.ib()
    n_gens = attr.ib()
    rho = attr.ib()
    L = attr.ib()
    n_loci = attr.ib()
    h5_out = attr.ib()


    def __attrs_post_init__(self):
        self.simulate()
        self.trace()


    def simulate(self):
        """
        Performs forward simulations with simuPOP, saving genotypes to file
        """
        self.FSim = fsim.ForwardSim(self.n_inds, self.L, self.n_gens,
                               self.n_loci, self.rho)
        self.FSim.evolve()
        self.FSim.write_haplotypes(self.h5_out)

        ## Initialize population and trace haplotype lineages
        self.ID = self.FSim.ID.ravel()
        self.recs = self.FSim.recs


    def trace(self):
        """
        Traces lineages through forward simulations
        """
        P = trace.Population(self.ID, self.recs, self.n_gens,
                             self.n_loci)
        P.trace()

        ## Convert traces haplotypes into an msprime TreeSequence
        self.positions = self.FSim.pop.lociPos()
        self.T = trace.TreeBuilder(P.haps, self.positions)
        self.ts = self.T.tree_sequence()


    def genotypes(self, nodes):
        """
        Returns the full genotypes of the provided nodes
        """
        return self.T.genotypes(nodes, self.h5_out)


    def draw_locus(self, locus, out_file='tree.svg'):
        """
        Draws the coalescent tree associated with the specified locus
        """
        for t in self.ts.trees():
            left, right = t.interval

            if left <= locus < right:
                t.draw(out_file, width=5000, height=500, show_times=True)
                break


def main(args):
    W = WFTree(**vars(args))

    return W


if __name__ == "__main__":
    args = argparse.Namespace(
            n_inds=10,
            n_gens=2,
            rho=1e-8,
            L=1e8,
            n_loci=20,
            h5_out='gen.h5'
            )

    W = main(args)
