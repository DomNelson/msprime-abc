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
    t_divergence = attr.ib()
    t_admixture = attr.ib()


    def __attrs_post_init__(self):
        self.nodes = list(self.TreeSequence.nodes())


    def ancestors(self, SparseTree):
        stack = [SparseTree.get_root()]
        while len(stack) > 0:
            v = stack.pop()
            ##TODO: Are all leaves time == 0?
            if self.nodes[v].time > self.t_admixture + 2:
                stack.extend(reversed(SparseTree.get_children(v)))
            elif self.nodes[v].time > self.t_admixture:
                yield v, self.nodes[v].population


    def get_ancestry_tracts(self, SparseTree):
        """
        Returns a dict containing the number of tracts of each ancestry
        contained in the provided tree, along with their length
        """
        length = SparseTree.interval[1] - SparseTree.interval[0]
        num_tracts = defaultdict(int)

        for ancestor, ancestry in self.ancestors(SparseTree):
            ## Each leaf descended from an ancestor represents a
            ## copy of this ancestry tract
            new_leaves = SparseTree.leaves(ancestor)

            ## Sum elements of generator
            num_tracts[ancestry] += np.sum([1 for _ in new_leaves])

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

# @profile
def main(args):
    ## Initialize admixed and source populations
    population_configurations = [
            msprime.PopulationConfiguration(sample_size=0,
                                        initial_size=args.Na*args.admixed_prop,
                                        growth_rate=0),
            msprime.PopulationConfiguration(sample_size=args.Na,
                                        growth_rate=0)]

    ## Specify admixture event
    demographic_events = [
            msprime.MassMigration(time=args.t_admix, source=1, destination=0,
                                    proportion=args.admixed_prop),
            msprime.PopulationParametersChange(time=args.t_admix+1,
                                    initial_size=1),
            msprime.MassMigration(time=args.t_admix+2, source=0, destination=1,
                                    proportion=1.),
            msprime.PopulationParametersChange(time=args.t_admix+2,
                                    initial_size=1)]
            
    ## Coalescent simulation
    ts = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            recombination_rate=1e-8, length=1e9, Ne=1)

    ta = TreeAncestry(ts, args.t_div, args.t_admix)

    print(ta.bin_ancestry_tracts(bins=20))
    # dp = msprime.DemographyDebugger(
    #     Ne=args.Na,
    #     population_configurations=population_configurations,
    #     demographic_events=demographic_events)
    # dp.print_history()


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
