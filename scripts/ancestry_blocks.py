import msprime
import numpy as np
import sys, os
import math
import attr
from collections import defaultdict, Counter

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
        self.records = self.TreeSequence.records()
        self.ancestors = self.get_ancestors()


    def get_ancestors(self):
        ## Collect nodes from the source populations
        ancestry_dict = {}
        for i, node in enumerate(ts.nodes()):
            if self.t_divergence > node.time and self.t_admixture < node.time:
                ancestry_dict[i] = node.population

        return ancestry_dict


    def get_ancestry_tracts(self, SparseTree):
        """
        Returns a dict containing the number of tracts of each ancestry
        contained in the provided tree, along with their length
        """
        length = SparseTree.interval[1] - SparseTree.interval[0]
        ancestry_tracts = defaultdict(int)

        for ancestor, ancestry in self.ancestors.items():
            try:
                if SparseTree.is_internal(ancestor):
                    ## Each leaf descended from an ancestor represents a
                    ## copy of this ancestry tract
                    num_copies = SparseTree.num_leaves(ancestor)
                    ancestry_tracts[ancestry] += num_copies
            except ValueError:
                ## Raised when node is not present in tree
                continue

        return ancestry_tracts, length
            

    def bin_ancestry_tracts(self, bins=None):
        """ Returns a histogram of tract lengths for each ancestry"""
        tract_lengths = defaultdict(Counter)
        tract_length_hist = {}

        for tree in self.TreeSequence.trees():
            ancestry_tracts, length = self.get_ancestry_tracts(tree)

            ## Update the count of tract lengths within each ancestry
            for ancestry, num_copies in ancestry_tracts.items():
                tract_lengths[ancestry] += {length: num_copies}

        ## Convert Counter object of tract lengths to histogram
        for ancestry, counts in tract_lengths.items():
            tract_length_hist[ancestry] = counter_to_hist(counts, bins)

        return tract_length_hist


admixed_sample_size = 300
admixed_pop_size = 10000
t_div = 1000
t_admix = 20
Ne = 10000

## Set parameters of admixture event
admixed_prop = 0.7
A_size = admixed_sample_size * admixed_prop
B_size = admixed_sample_size * 1 - admixed_prop

## Initialize admixed and source populations
population_configurations = [
        msprime.PopulationConfiguration(sample_size=0,
                                        initial_size=Ne,
                                        growth_rate=0),
        msprime.PopulationConfiguration(sample_size=admixed_sample_size,
                                        growth_rate=0)]

## Specify admixture event
demographic_events = [
        msprime.MassMigration(time=t_admix, source=1, destination=0,
                                proportion=admixed_prop),
        msprime.MassMigration(time=t_div, source=0, destination=1,
                                proportion=1.)]
        
dp = msprime.DemographyDebugger(
    Ne=Ne,
    population_configurations=population_configurations,
    demographic_events=demographic_events)
dp.print_history()

## Coalescent simulation
ts = msprime.simulate(population_configurations=population_configurations,
                        demographic_events=demographic_events,
                        recombination_rate=1e-8, length=1e5, Ne=Ne)

ta = TreeAncestry(ts, t_div, t_admix)
print(ta.bin_ancestry_tracts(bins=20))
