import msprime
import numpy as np
import sys, os
import math
import attr
from collections import defaultdict, Counter
import argparse
from profilehooks import profile


def counter_to_hist(counter, bins=None):
    """ Converts a Counter object to a numpy histogram """
    ##TODO: Could do this without expanding the list of items
    return np.histogram(list(counter.elements()), bins=bins)


@attr.s
class TreeAncestry:
    TreeSequence = attr.ib()
    t_admixture = attr.ib()


    def __attrs_post_init__(self):
        self.nodes = list(self.TreeSequence.nodes())
        ## Containers holding ancestry tracts as they're constructed, and once
        ## they are completed
        self.current_tracts = defaultdict(dict)
        self.tract_lengths = defaultdict(Counter)


    def ancestors(self, SparseTree):
        stack = [SparseTree.get_root()]
        while len(stack) > 0:
            v = stack.pop()
            ##TODO: Are all leaves time == 0?
            if self.nodes[v].time > self.t_admixture+5:
                stack.extend(reversed(SparseTree.get_children(v)))
            elif self.nodes[v].time > self.t_admixture:
                yield v, self.nodes[v].population


    def leaf_ancestries(self, SparseTree):
        """ Returns a dict of leaves belonging to each ancestry """
        leaf_set = defaultdict(set)

        for ancestor, ancestry in self.ancestors(SparseTree):
            leaf_set[ancestry].update(SparseTree.leaves(ancestor))

        return leaf_set


    def start_new_tracts(self, length, leaf_set, ancestry):
        """
        Initializes new tracts for leaves which have just switched ancestry
        """
        current_leaves = set(self.current_tracts[ancestry].keys())
        new_leaves = leaf_set.difference(current_leaves)

        start_tracts = {t:length for t in new_leaves}
        self.current_tracts[ancestry].update(start_tracts)


    def increment_tracts(self, length, leaf_set, ancestry):
        """ Extends tracts of leaves which do not change ancestry """
        current_leaves = set(self.current_tracts[ancestry].keys())
        for leaf in current_leaves.intersection(leaf_set):
            self.current_tracts[ancestry][leaf] += length


    def add_complete_tracts(self, leaf_set, ancestry):
        """ 
        Stores complete tracts for leaves which have jsut switched ancestry
        """
        current_leaves = set(self.current_tracts[ancestry].keys())
        switch_leaves = current_leaves.difference(leaf_set)

        lengths = []
        for leaf in switch_leaves:
            lengths.append(self.current_tracts[ancestry].pop(leaf))

        self.tract_lengths[ancestry].update(lengths)


    def set_tract_lengths(self):
        """ Returns a Counter object of tract lengths for each ancestry """
        for tree in self.TreeSequence.trees(leaf_lists=True):
            ## Get the ancestries of each leaf in the tree
            leaf_ancestries = self.leaf_ancestries(tree)

            for ancestry, leaf_set in leaf_ancestries.items():
                ## Find tracts which start in this tree and assign them the
                ## length of the tree, possibly to be incremented later
                self.start_new_tracts(tree.length, leaf_set, ancestry)

                ## Leaves which switch ancestry mark the end of a tract
                self.add_complete_tracts(leaf_set, ancestry)

                ## Leaves which remain have their tract lengths incremented
                self.increment_tracts(tree.length, leaf_set, ancestry)


    def bin_ancestry_tracts(self, bins=None):
        """ Returns a histogram of tract lengths for each ancestry"""
        tract_length_hist = {}
        ancestry_length = {}

        ## Convert Counter object of tract lengths to histogram
        for ancestry, counts in self.tract_lengths.items():
            tract_length_hist[ancestry] = counter_to_hist(counts, bins)
            ancestry_length[ancestry] = sum(counts.elements())

        return tract_length_hist, ancestry_length

# @profile
def main(args):
    ## Initialize admixed and source populations
    population_configurations = [
            msprime.PopulationConfiguration(sample_size=0,
                                            initial_size=args.Na,
                                            growth_rate=0),
            msprime.PopulationConfiguration(sample_size=args.Na,
                                            growth_rate=0)]

    ## Specify admixture event
    demographic_events = [
            msprime.MassMigration(time=args.t_admix, source=1, destination=0,
                                    proportion=args.admixed_prop),
            msprime.PopulationParametersChange(time=args.t_admix+1,
                                    initial_size=1),
            msprime.MassMigration(time=args.t_admix+5, source=0, destination=1,
                                    proportion=1.),
            msprime.PopulationParametersChange(time=args.t_admix+5,
                                    initial_size=1)]
            
    ## Coalescent simulation
    ts = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            recombination_rate=args.rho, length=args.length,
                            Ne=args.Na)

    ta = TreeAncestry(ts, args.t_admix)
    ta.set_tract_lengths()

    tracts, ancestry_length = ta.bin_ancestry_tracts(bins=20)
    ancestry_props = {a: l / (args.Na * args.length)
                      for a, l in ancestry_length.items()}

    return tracts, ancestry_props


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Na", help="Size of admixed population",
            required=True, type=int)
    parser.add_argument("--t_admix", help="Time of admixture event",
            required=True, type=int)
    parser.add_argument("--admixed_prop", help="Admixture proportion of pop 1",
            required=True, type=float)
    parser.add_argument("--length", help="Length in base pairs to simulate",
            required=True, type=float)
    parser.add_argument("--rho", help="Recombination rate per base pair, " +\
            "default=1e-8", type=float, default=1e-8)

    args = parser.parse_args()

    ## Sanity check args
    assert 0 <= args.admixed_prop <= 1, "Invalid admixture proportion"

    main(args)
