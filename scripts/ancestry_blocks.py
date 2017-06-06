import msprime
import numpy as np
import sys, os
import math
import attr
from collections import defaultdict, Counter
import argparse
from profilehooks import profile


def get_bin_edges(length, nbins):
    bin_width = length / nbins
    bin_edges = np.concatenate([np.arange(0, length, bin_width), [length]])

    return bin_edges


@attr.s
class TractLengths:
    TreeSequence = attr.ib()
    t_admix = attr.ib()


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
            ##TODO: Are all leaves time == 0? +p2
            if self.nodes[v].time == self.t_admix + 3:
                stack.extend(reversed(SparseTree.get_children(v)))
            elif self.nodes[v].time > self.t_admix:
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
                ##TODO: More efficient to do all these steps at once? +p3
                self.start_new_tracts(tree.length, leaf_set, ancestry)

                ## Leaves which switch ancestry mark the end of a tract
                self.add_complete_tracts(leaf_set, ancestry)

                ## Leaves which remain have their tract lengths incremented
                self.increment_tracts(tree.length, leaf_set, ancestry)


def counter_to_hist(counter, bins=None):
    """ Converts a Counter object to a numpy histogram """
    ##TODO: Could do this without expanding the list of items +p3
    return np.histogram(list(counter.elements()), bins=bins)


def bin_ancestry_tracts(tract_lengths, Na, length, bins=None):
    """ Returns a histogram of tract lengths for each ancestry"""
    ## Default value is an empty histogram, if one ancestry has no
    ## tracts in the leaves
    empty_hist = lambda: np.histogram([], bins=bins)
    empty_ancestry_prop = lambda: 0

    tract_length_hist = defaultdict(empty_hist)
    ancestry_props = defaultdict(empty_ancestry_prop)

    ## Convert Counter object of tract lengths to histogram
    for ancestry, counts in tract_lengths.items():
        tract_length_hist[ancestry] = counter_to_hist(counts, bins)
        ancestry_length = sum(counts.elements()) / Na
        ancestry_props[ancestry] = ancestry_length / length

    return tract_length_hist, ancestry_props


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
            msprime.SimpleBottleneck(time=args.t_admix+1, population_id=0,
                                    proportion=1.0),
            msprime.SimpleBottleneck(time=args.t_admix+1, population_id=1,
                                    proportion=1.0),
            msprime.MassMigration(time=args.t_admix+2, source=0, destination=1,
                                    proportion=1.),
            msprime.SimpleBottleneck(time=args.t_admix+3, population_id=1,
                                    proportion=1.0)]
            
    ## Coalescent simulation
    ts = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            recombination_rate=args.rho, length=args.length,
                            Ne=args.Na)

    tl = TractLengths(ts, args.t_admix)
    tl.set_tract_lengths()

    ## Create histogram of ancestry tract lengths
    bin_edges = get_bin_edges(args.length, args.nbins)
    tracts, ancestry_props = bin_ancestry_tracts(
            tl.tract_lengths,
            args.Na,
            args.length,
            bins=bin_edges)

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
    parser.add_argument("--nbins", help="Number of bins in tract-length " +\
            "histogram. Default 10", type=int, default=10)

    args = parser.parse_args()

    ## Sanity check args
    assert 0 <= args.admixed_prop <= 1, "Invalid admixture proportion"

    main(args)
