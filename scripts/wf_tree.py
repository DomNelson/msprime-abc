import numpy as np
import sys, os
from profilehooks import timecall
import argparse
import configparser
import attr

import wf_sims
import pop_models
import trace_tree


@attr.s
class WFTree(object):
    simulator = attr.ib()
    h5_out = attr.ib()
    discrete_loci = attr.ib(default=True)
    output = attr.ib(default='genotypes.txt')
    sample_size = attr.ib(default='all')
    tracked_loci = attr.ib(default=False)


    def __attrs_post_init__(self):
        self.verify_simulator()
        self.trace()
        self.set_tree_sequence()


    def verify_simulator(self):
        """
        Verifies that the provided simulator has the required attributes
        and methods to be used
        """
        assert hasattr(self.simulator, 'L'), \
                "Simulator must expose genome length as L"
        assert not callable(self.simulator.init_IDs), \
                "Simulator must expose an iterable of initial node labels"
        assert type(self.simulator.discrete_loci) is bool, \
                "Simulator must specify whether loci are indices or positions"
        assert callable(self.simulator.draw_recomb_vals), \
                "Simulator must impement a method for drawing recombinations"


    def trace(self):
        """
        Traces lineages through forward simulations
        """
        ## Set number of loci from initial population
        ##NOTE: Assumes a single chromosome +n2
        self.P = trace_tree.Population(sample_size=self.sample_size,
                                   discrete_loci=self.simulator.discrete_loci)
        self.P.init_haps(self.simulator.init_IDs, self.simulator.L)
        self.P.trace(self.simulator.draw_recomb_vals())


    def set_tree_sequence(self, pop_dict=None):
        ## Convert traces haplotypes into an msprime TreeSequence
        self.T = trace_tree.TreeBuilder(self.P.haps, pop_dict)
        self.ts = self.T.ts


    def genotypes(self, nodes):
        """
        Returns the full genotypes of the provided nodes
        """
        return self.T.genotypes(nodes, self.h5_out)


    def mutate(self):
        """
        Thrown down mutations on constructed tree sequence, replacing
        self.ts with new mutated (but otherwise identical) tree sequence
        """
        self.ts = trace_tree.mutate_ts(self.ts, self.mu)


    def _mutations(self):
        """
        Return mutation events as a structured numpy array
        """
        ##TODO: Nothing here until we implement selection +t1
        muts = self.FSim.muts

        ## Convert simuPOP IDs into their corresponding tree node IDs
        ts_IDs = [self.T.haps_idx[sim_ID] for sim_ID in muts['ID']]

        muts['ID'] = ts_IDs

        return muts


    def plot_locus(self, locus, out_file='tree.svg'):
        """
        Draws the coalescent tree associated with the specified locus
        """
        for t in self.ts.trees():
            left, right = t.interval

            if left <= locus < right:
                t.draw(out_file, width=500, height=500, show_times=True)
                break


def example_simulators(pop_type='simple'):
    if pop_type == 'backwards':
        B = wf_sims.BackwardSim(args.n_gens, args.n_inds, args.L, args.rho)

        return B

    if pop_type == 'simple':
        msp_pop = pop_models.simple_pop_ts(args.n_inds, args.rho, args.L,
                                           args.mu, args.Ne, args.ploidy)
    elif pop_type == 'grid':
        msp_pop = pop_models.grid_ts(N=args.n_inds*args.ploidy,
                        rho=args.rho, L=args.L, mu=args.mu, t_div=args.t_div,
                        Ne=args.Ne, mig_prob=args.mig_prob,
                        grid_width=args.grid_width)

    simuPOP_pop = pop_models.msp_to_simuPOP(msp_pop)
    F = wf_sims.ForwardSim(args.n_gens, simuPOP_pop)
    F.evolve()

    return F


def main(args):
    simulator = example_simulators('backwards')

    W = WFTree(simulator=simulator,
               h5_out=args.h5_out,
               tracked_loci=args.tracked_loci)

    return W


if __name__ == "__main__":
    args = argparse.Namespace(
            n_inds=100,
            Ne=100,
            n_gens=1000,
            rho=1e-8,
            mu=1e-8,
            L=1e7,
            mig_prob=0.25,
            grid_width=2,
            t_div=100,
            h5_out='gen.h5',
            MAF=0.1,
            ploidy=2,
            tracked_loci=True
            )

    W = main(args)
