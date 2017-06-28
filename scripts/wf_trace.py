import numpy as np
from profilehooks import timecall
import attr
from collections import defaultdict


def region_overap(regions):
    """
    Returns a dict with format {(*old_regions_ix,): new_region} where
    each new_region has a unique coalescence pattern
    """
    ##TODO This will create non-contiguous regions if disjoint segments are shared +t1
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


    def __attrs_post_init__(self):
        assert self.node >= 0
        assert len(self.loci) > 0
        assert set(self.loci) == set(range(self.loci[0], self.loci[-1]+1))


    def climb(self, node):
        """ Climbs to the specified node """
        return Haplotype(node, self.loci, self.children)


    def split(self, loci_ix):
        """
        Returns new haplotypes, representing the current haplotype split at
        the given loci
        """
        ## Shift split points to be relative to the start of the current
        ## segment, and create first segment
        loci_ix -= self.loci[0]
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
    ID = attr.ib()
    lineage = attr.ib()
    recs = attr.ib()
    n_gens = attr.ib()


    def __attrs_post_init__(self):
        self.haps = self.init_haps()


    def init_haps(self):
        """
        Initializes haplotypes according to ID labels, in last generation,
        with loci numbered sequentially
        """
        _, n_loci = self.lineage.shape
        self.n_inds = int(self.lineage.shape[0] / self.n_gens)

        nodes = self.ID[-self.n_inds:]
        loci = tuple(np.arange(n_loci))
        haps = set([Haplotype(n, loci, [n]) for n in nodes])

        return haps


    def active_haps(self):
        """ Returns current active lineages in the population """
        return [h for h in self.haps if h.active is True]


    def collect_active_haps(self):
        """ Returns haplotypes collected by current node """
        nodes = defaultdict(list)
        
        for hap in self.active_haps():
            nodes[hap.node].append(hap)

        return nodes


    def detailed_lineage(self):
        """
        For debugging, shows ID, lineage, and recombinations side-by-side
        """
        return np.hstack([self.ID.reshape(-1, 1),
                          self.lineage,
                          np.array(self.recs).reshape(-1, 1)])


    def recombine(self):
        """
        Climbs haplotypes to the appropriate parent node, splitting them
        if recombination occurred
        """
        ##TODO Check if these lists are necessary +t3
        new_haps = []
        old_haps = []
        for hap in self.active_haps():
            node = hap.node
            ## Format is [Offspring, Parent, StartChrom, rec1, rec2, ...]
            recs = self.recs[node][3:]

            ## Only split if both sides of breakpoint are in Haplotype
            recs = np.array([r for r in recs
                                if (r in hap.loci and r+1 in hap.loci)])

            ## Replace old haplotype with new split
            if len(recs) > 0:
                new_haps.extend(hap.split(recs))
                old_haps.append(hap)

        self.haps.update(new_haps)
        for h in old_haps:
            assert h in self.haps
            self.haps.discard(h)


    def climb(self):
        """ Climb all haplotypes to their parent node """
        for hap in self.active_haps():
            parent_ix = hap.loci[0]
            new_node = self.lineage[hap.node][parent_ix]

            ## Negative nodes indicate the top of the lineage
            if new_node < 0:
                self.haps.add(Haplotype(hap.node, hap.loci, hap.children,
                                active=False))
            else:
                self.haps.add(hap.climb(new_node))
            self.haps.discard(hap)


    def coalesce(self):
        """
        Coalesces haplotypes that share loci within a common ancestor
        """
        for node, haps in self.collect_active_haps().items():

            if len(haps) > 1:
                regions = [h.loci for h in haps]
                new_regions = region_overap(regions)

                for old_regions, new_region in new_regions:
                    ## If there is only one old_region, then no coalescence
                    ## happened in this segment and it stays active
                    active = (len(old_regions) == 1)

                    ## Create coalesced haplotypes
                    children = []
                    for i in old_regions:
                        children.extend(haps[i].children)
                    children = tuple(sorted(children))

                    ## Create an inactive haplotype recording the
                    ## coalescence
                    coalesced_hap = Haplotype(node, new_region, children,
                                              active=active)
                    self.haps.add(coalesced_hap)

                    ## If coalescence happened, create a new haplotype to
                    ## continue climbing
                    if active is False:
                        self.haps.add(Haplotype(node, new_region, [node]))

                ## Remove old haplotypes
                for h in haps:
                    self.haps.discard(h)


    @timecall
    def trace(self):
        """ Traces coalescent events through the population lineage """
        while len(self.active_haps()) > 0:
            self.recombine()
            self.climb()
            self.coalesce()
