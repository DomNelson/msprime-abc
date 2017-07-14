import numpy as np
import sys, os
from profilehooks import timecall
import argparse
import configparser
import attr

import forward_sim as fsim
import pop_models
import trace_tree


@attr.s
class WFTree(object):
    n_inds = attr.ib()
    n_gens = attr.ib()
    rho = attr.ib()
    L = attr.ib()
    # n_loci = attr.ib() # Set from msprime initial pop
    h5_out = attr.ib()
    mu = attr.ib()
    MAF = attr.ib()
    trace_trees = attr.ib(default=True)
    Ne = attr.ib(default=1000)
    sample_size = attr.ib(default='all')
    save_genotypes = attr.ib(default=True)
    t_div = attr.ib(default=100)
    mig_prob = attr.ib(default=0)
    grid_width = attr.ib(default=3)
    ploidy = attr.ib(default=2)


    def __attrs_post_init__(self):
        self.simulate()

        if self.trace_trees is True:
            self.trace()


    def simulate(self):
        """
        Performs forward simulations with simuPOP, saving genotypes to file
        """
        ## Create initial population
        # self.msprime_pop = pop_models.grid_ts(N=self.n_inds*self.ploidy,
        #                 rho=self.rho,
        #                 L=self.L, mu=self.mu, t_div=self.t_div, Ne=self.Ne,
        #                 mig_prob=self.mig_prob, grid_width=self.grid_width)
        #
        # init_pop = pop_models.msp_to_simuPOP(self.msprime_pop)

        ## Initialize grid of demes with a single locus
        N = np.array([args.n_inds for i in range(self.grid_width**2)])
        MAF = np.array([0.2 for i in range(self.grid_width**2)]).reshape(-1, 1)
        migmat = pop_models.grid_migration(self.grid_width, 0.1)

        init_pop = pop_models.maf_init_simuPOP(N, self.rho, self.L, self.mu,
                                    MAF, migmat=migmat)

        ##TODO: Implement mutations in forward sims +t1
        self.FSim = fsim.ForwardSim(self.n_gens, init_pop)
        self.FSim.evolve(self.save_genotypes)

        if self.save_genotypes is True:
            self.FSim.write_haplotypes(self.h5_out)

        ## Initialize population and trace haplotype lineages
        self.ID = self.FSim.ID.ravel()
        self.recs = self.FSim.recs


    def trace(self):
        """
        Traces lineages through forward simulations
        """
        ## Set number of loci from initial population
        ##NOTE: Assumes a single chromosome +n2
        n_loci = self.FSim.pop.numLoci()[0]
        self.P = trace_tree.Population(self.ID, self.recs, self.n_gens, n_loci,
                                        sample_size=self.sample_size)
        self.P.trace()

        ## Convert traces haplotypes into an msprime TreeSequence
        self.positions = self.FSim.pop.lociPos()
        self.T = trace_tree.TreeBuilder(self.P.haps, self.positions)
        self.ts = self.T.tree_sequence()


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


def main(args):
    W = WFTree(**vars(args))

    return W


if __name__ == "__main__":
    args = argparse.Namespace(
            n_inds=200,
            Ne=100,
            n_gens=10,
            rho=1e-8,
            mu=1e-10,
            L=1e8,
            mig_prob=0.05,
            # n_loci=20,
            h5_out='gen.h5',
            MAF=0.1,
            save_genotypes=False
            )

    W = main(args)
