import numpy as np
import attr
from collections import defaultdict, Counter


def region_overap(regions):
    """
    Returns a dict with format {(*old_regions_ix,): new_region} where
    each new_region has a unique coalescence pattern
    """
    points = set().union(*regions)
    new_regions = defaultdict(list)

    for pt in points:
        next_regions = tuple([i for i, r in enumerate(regions) if pt in r])
        new_regions[next_regions].append(pt)

    for old_regions, new_region in new_regions.items():

            yield old_regions, tuple(new_region)

        

@attr.s(frozen=True)
class Haplotype(object):
    node = attr.ib(convert=int)
    loci = attr.ib(convert=tuple)
    children = attr.ib(convert=tuple)
    active = attr.ib(default=True)


    # def __eq__(self, other):
    #     return self.__dict__ == other.__dict__
    #
    #
    # def __hash__(self):
    #     return hash(self.node, self.loci, self.children, self.active)


    def climb(self, node):
        """
        Climbs to the specified node
        """
        return Haplotype(node, self.loci, self.children)


    def split(self, locus_ix):
        """
        Truncates self at locus and returns a new instance with the remainder,
        which is still associated with the original node
        """
        left_loci = self.loci[locus_ix:]
        right_loci = self.loci[:locus_ix]

        left_hap = Haplotype(self.node, left_loci, self.children)
        right_hap = Haplotype(self.node, right_loci, self.children)

        return left_hap, right_hap


@attr.s
class Population(object):
    """
    A population of haplotypes, with methods for retracing coalescent trees
    through a lineage constructed by forward simulations
    """
    fsim = attr.ib()


    def __attrs_post_init__(self):
        self.ID = self.fsim.get_idx(self.fsim.ID).ravel()
        self.lineage = self.fsim.get_idx(self.fsim.lineage)
        self.recs = self.fsim.recs
        self.n_chroms, self.n_loci = self.fsim.genotype.shape
        self.n_gens = self.fsim.n_gens
        self.n_inds = int(self.lineage.shape[0] / self.n_gens)

        self.haps = self.init_haps(self.fsim)


    def init_haps(self, fsim):
        nodes = self.ID[-self.n_inds:]
        loci = tuple(np.arange(self.n_loci))
        haps = set([Haplotype(n, loci, [n]) for n in nodes])

        return haps


    def haps_by_state(self, state):
        """
        Returns current active lineages in the population
        """
        active_haps = [h for h in self.haps if h.active is state]
        print("Active")
        for hap in active_haps:
            print(hap in self.haps)

        return active_haps


    def collect_active_haps(self):
        """
        Returns haplotypes collected by current node
        """
        nodes = defaultdict(list)
        print("Collecting")
        
        for hap in self.haps_by_state(True):
            print(hap in self.haps)
            nodes[hap.node].append(hap)

        return nodes


    def detailed_lineage(self):
        """
        For debugging, shows ID, lineage, and recombinations side-by-side
        """
        return np.hstack([self.ID[self.n_inds:].reshape(-1, 1),
                          self.lineage[self.n_inds:],
                          np.array(self.recs[self.n_inds:]).reshape(-1, 1)])


    def recombine(self):
        """
        Climbs haplotypes to the appropriate parent node, splitting them
        if recombination occurred
        """
        new_haps = []
        old_haps = []
        for hap in self.haps:
            node = hap.node
            recs = self.recs[node]

            ## Format is [Offspring, Parent, StartChrom, rec1, rec2, ...]
            ## so recs[3:] is an empty list if no recombinations occurred
            for rec in recs[3:]:
                ## Get nodes on either side of the split, and make sure
                ## a recombination actually occurred
                if rec in hap.loci:
                    new_nodes = self.lineage[node][rec: rec + 2]
                    assert len(set(new_nodes)) == 2

                    ## Replace old haplotype with new split
                    new_haps.extend(hap.split(rec+1))

        self.haps.update(new_haps)
        for h in old_haps:
            assert h in self.haps
            self.haps.discard(h)


    def climb(self):
        """
        Climb all haplotypes to their parent node
        """
        active_haps = self.haps_by_state(True)
        for hap in active_haps:
            parent_ix = hap.loci[0]
            new_node = self.lineage[hap.node][parent_ix]
            self.haps.add(hap.climb(new_node))
            assert hap in self.haps
            self.haps.discard(hap)


    def coalesce(self):
        """
        Coalesces haplotypes that share loci within a common ancestor
        """
        for node, haps in self.collect_active_haps().items():
            print("-" * 60)

            if len(haps) > 1:
                print("Coalescing", haps)
                regions = [h.loci for h in haps]
                new_regions = region_overap(regions)

                for old_regions, new_region in new_regions:
                    ## Modify loci associated with uncoalesced regions
                    if len(old_regions) == 1:
                        old_hap = haps[old_regions[0]]
                        new_hap = Haplotype(old_hap.node, new_region,
                                            old_hap.children)

                        ## Replace haplotype with updated region
                        self.haps.discard(old_hap)
                        self.haps.add(new_hap)

                    else:
                    ## Create coalesced haplotypes
                        children = []
                        for i in old_regions:
                            children.extend(haps[i].children)
                        children = tuple(sorted(children))

                        ## Create an inactive haplotype recording the
                        ## coalescence
                        coalesced_hap = Haplotype(node,
                                                  new_region,
                                                  children,
                                                  active=False)
                        self.haps.add(coalesced_hap)
                        print("\nCoalesced hap", coalesced_hap)

                        ## Create a new haplotype to continue climbing
                        new_hap = Haplotype(node, new_region, [node])
                        self.haps.add(new_hap)
                        print("New hap", new_hap, "\n")

                        ## Remove old haplotypes
                        for i in old_regions:
                            print("Removing", haps[i])
                            print(haps[i] in self.haps)
                            self.haps.discard(haps[i])
                            print(haps[i] in self.haps)


@attr.s
class ForwardTree(object):
    fsim = attr.ib()
    records = attr.ib()
    haplotypes = attr.ib()


    def climb(self):
        """
        Ascends the lineages of the given haplotypes until a recombination
        occurs
        """
        ancs = Counter(self.lineage)
