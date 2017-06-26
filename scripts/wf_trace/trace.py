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


    def climb(self, node):
        """
        Climbs to the specified node
        """
        return Haplotype(node, self.loci, self.children)


    def split(self, loci_ix):
        """
        Returns new haplotypes, representing the current haplotype split at
        the given loci
        """
        segments = [self.loci[:loci_ix[0]+1]]

        for i in range(1, len(loci_ix)):
            segments.append(self.loci[loci_ix[i-1]+1:loci_ix[i]+1])

        ## Add the last segment
        segments.append(self.loci[loci_ix[-1]+1:])

        return [Haplotype(self.node, s, self.children) for s in segments]


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


    def active_haps(self):
        """
        Returns current active lineages in the population
        """
        return [h for h in self.haps if h.active is True]


    def collect_active_haps(self):
        """
        Returns haplotypes collected by current node
        """
        nodes = defaultdict(list)
        
        for hap in self.active_haps():
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
        for hap in self.active_haps():
            node = hap.node
            ## Format is [Offspring, Parent, StartChrom, rec1, rec2, ...]
            recs = self.recs[node][3:]

            ## Only split if both sides of breakpoint are in Haplotype
            recs = [r for r in recs if (r in hap.loci and r+1 in hap.loci)]

            ## Replace old haplotype with new split
            if len(recs) > 0:
                new_haps.extend(hap.split(recs))
                old_haps.append(hap)

        self.haps.update(new_haps)
        for h in old_haps:
            assert h in self.haps
            self.haps.discard(h)


    def climb(self):
        """
        Climb all haplotypes to their parent node
        """
        for hap in self.active_haps():
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
                for h in haps:
                    print("Removing", h)
                    print(h in self.haps)
                    self.haps.discard(h)
                    print(h in self.haps)


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
