import msprime
import numpy as np
import sys, os
import math
import attr
from collections import defaultdict, Counter
import argparse


def counter_to_hist(counter, bins=None):
    """ Converts a Counter object to a numpy histogram """
    ##TODO: Could do this without expanding the list of items
    return np.histogram(list(counter.elements()), bins=bins)


@attr.s
class TreeAncestry:
    TreeSequence = attr.ib()
    t_divergence = attr.ib()
    t_admixture = attr.ib()

    def __attrs_post_init__(self):
        self.ancestors = self.get_ancestors()


    def get_ancestors(self):
        """
        Collect nodes from the diverged populations which formed the
        admixed sample
        """
        ancestry_dict = {}
        for i, node in enumerate(self.TreeSequence.nodes()):
            if self.t_divergence > node.time and self.t_admixture < node.time:
                ancestry_dict[i] = node.population

        return ancestry_dict


    def get_ancestry_tracts(self, SparseTree):
        """
        Returns a dict containing the number of tracts of each ancestry
        contained in the provided tree, along with their length
        """
        length = SparseTree.interval[1] - SparseTree.interval[0]
        ancestry_leaves = defaultdict(set)

        for ancestor, ancestry in self.ancestors.items():
            ##TODO: Is there a better was of checking if node is in a tree?
            try:
                if SparseTree.is_internal(ancestor):
                    ## Each leaf descended from an ancestor represents a
                    ## copy of this ancestry tract
                    new_leaves = set(SparseTree.leaves(ancestor))
                    ancestry_leaves[ancestry].update(new_leaves)
            except ValueError:
                ## Raised when node is not present in tree
                continue

        ## Get number of unique copies of each ancestry tract
        num_tracts = {a: len(l) for a, l in ancestry_leaves.items()}

        return num_tracts, length
            

    def bin_ancestry_tracts(self, bins=None):
        """ Returns a histogram of tract lengths for each ancestry"""
        tract_lengths = defaultdict(Counter)
        tract_length_hist = {}

        ##TODO: Does setting leaf_lists=True help here?
        for tree in self.TreeSequence.trees(leaf_lists=True):
            ancestry_tracts, length = self.get_ancestry_tracts(tree)

            ## Update the count of tract lengths within each ancestry
            for ancestry, num_copies in ancestry_tracts.items():
                tract_lengths[ancestry] += {length: num_copies}

        ## Convert Counter object of tract lengths to histogram
        for ancestry, counts in tract_lengths.items():
            tract_length_hist[ancestry] = counter_to_hist(counts, bins)

        return tract_length_hist


def main(args):
    ## Initialize admixed and source populations
    population_configurations = [
            msprime.PopulationConfiguration(sample_size=0,
                                            initial_size=args.Ne,
                                            growth_rate=0),
            msprime.PopulationConfiguration(sample_size=args.Na,
                                            growth_rate=0)]

    ## Specify admixture event
    demographic_events = [
            msprime.MassMigration(time=args.t_admix, source=1, destination=0,
                                    proportion=args.admixed_prop),
            msprime.MassMigration(time=args.t_div, source=0, destination=1,
                                    proportion=1.)]
            
    ## Coalescent simulation
    ts = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            recombination_rate=1e-8, length=1e5, Ne=args.Ne)

    ta = TreeAncestry(ts, args.t_div, args.t_admix)

    from IPython import embed; embed()

    print(ta.bin_ancestry_tracts(bins=20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Na", help="Size of admixed population",
            required=True, type=int)
    parser.add_argument("--Ne", help="Size of effective population",
            type=int, default=10000)
    parser.add_argument("--t_div", help="Time of divergence between source" +\
            " populations", required=True, type=int)
    parser.add_argument("--t_admix", help="Time of admixture event",
            required=True, type=int)
    parser.add_argument("--admixed_prop", help="Admixture proportion of pop 1",
            required=True, type=float)

    args = parser.parse_args()

    ## Sanity check args
    assert 0 <= args.admixed_prop <= 1, "Invalid admixture proportion"

    main(args)
