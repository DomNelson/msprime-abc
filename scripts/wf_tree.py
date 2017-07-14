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
class InitialPop(object):
    n_inds = attr.ib()
    rho = attr.ib()
    L = attr.ib()
    mu = attr.ib()
    t_div = attr.ib(default=100)
    mig_prob = attr.ib(default=0)
    grid_width = attr.ib(default=3)
    Ne = attr.ib(default=1000)
    ploidy = attr.ib(default=2)


    def __attrs_post_init__(self):
        self.pop = self.init_pop()


    def init_pop(self):
        """ Initialized population for forward simulations """
        ## Create initial population
        # self.msprime_pop = pop_models.grid_ts(N=self.n_inds*self.ploidy,
        #                 rho=self.rho,
        #                 L=self.L, mu=self.mu, t_div=self.t_div, Ne=self.Ne,
        #                 mig_prob=self.mig_prob, grid_width=self.grid_width)
        #
        # initial_pop = pop_models.msp_to_simuPOP(self.msprime_pop)

        ## Initialize grid of demes with a single locus
        N = np.array([self.n_inds for i in range(self.grid_width**2)])
        MAF = np.array([0.2 for i in range(self.grid_width**2)]).reshape(-1, 1)
        migmat = pop_models.grid_migration(self.grid_width, 0.1)

        initial_pop = pop_models.maf_init_simuPOP(N, self.rho, self.L, self.mu,
                                    MAF, migmat=migmat)

        return initial_pop


@attr.s
class WFTree(object):
    initial_pop = attr.ib()
    n_gens = attr.ib()
    h5_out = attr.ib()
    trace_trees = attr.ib(default=True)
    sample_size = attr.ib(default='all')
    save_genotypes = attr.ib(default=True)


    def __attrs_post_init__(self):
        self.simulate()

        if self.trace_trees is True:
            self.trace()


    def simulate(self):
        """
        Performs forward simulations with simuPOP, saving genotypes to file
        """
        self.FSim = fsim.ForwardSim(self.n_gens, self.initial_pop)
        self.FSim.evolve(self.save_genotypes)

        ##TODO: Save directly to hdf5 file +t2
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

        ## Pass population of each individual, setting populaiton of
        ## artificial root individual 0 to -1
        pop_dict = dict(self.FSim.subpops.tolist())

        ## Convert traces haplotypes into an msprime TreeSequence
        self.positions = self.FSim.pop.lociPos()
        self.T = trace_tree.TreeBuilder(self.P.haps, self.positions, pop_dict)
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


def main(args):
    initial_pop = InitialPop(
            n_inds=args.n_inds,
            rho=args.rho,
            L=args.L,
            mu=args.mu)

    W = WFTree(
            initial_pop=initial_pop.pop,
            n_gens=args.n_gens,
            h5_out=args.h5_out,
            save_genotypes=args.save_genotypes)

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