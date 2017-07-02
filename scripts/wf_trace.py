import numpy as np
import copy
import msprime
from profilehooks import timecall
import attr
from collections import defaultdict


def split_consecutive(data):
    """ Splits an array into runs of consecutive values """
    data = np.sort(data)
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)


def region_overlap(regions):
    """
    Returns a dict with format {(*old_regions_ix,): new_region} where
    each new_region has a unique coalescence pattern
    """
    points = sorted(set().union(*regions))

    for i in range(len(points)-1):
        old_regions = tuple([j for j, r in enumerate(regions)
                              if r[0] <= points[i] < r[1]])
        new_regions = (points[i], points[i+1])

        ## No old_regions indicates disjoint regions
        if len(old_regions) > 0:
            yield old_regions, new_regions


@attr.s(frozen=True)
class Haplotype(object):
    node = attr.ib(convert=int)
    left = attr.ib(convert=int)
    right = attr.ib(convert=int)
    children = attr.ib(convert=tuple)
    active = attr.ib(default=True)


    def __attrs_post_init__(self):
        assert self.node >= 0
        assert len(self.children) > 0
        assert self.left < self.right


    def climb(self, node):
        """ Climbs to the specified node """
        return Haplotype(node, self.left, self.right, self.children)


    def split(self, loci_ix):
        """
        Returns new haplotypes, representing the current haplotype split at
        the given loci
        """
        ## Shift split points to be relative to the start of the current
        ## segment, and create first segment
        loci_ix = np.array(loci_ix)
        assert ((loci_ix >= self.left) & (loci_ix < self.right)).all()

        ## Make sure we don't duplicate points if loci_ix contains either
        ## self.left or self.right - 1
        left_pts = sorted(set([self.left] + list(loci_ix + 1)))
        right_pts = sorted(set(list(loci_ix + 1) + [self.right]))

        for left, right in zip(left_pts, right_pts):
            yield Haplotype(self.node, left, right, self.children)


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

        ##NOTE Assumes constant generation size +n1
        times = np.arange(len(self.ID)) / self.n_gens
        self.times = np.floor(times)[::-1].astype(int)


    def init_haps(self):
        """
        Initializes haplotypes according to ID labels, in last generation,
        with loci numbered sequentially
        """
        _, n_loci = self.lineage.shape
        self.n_inds = int(self.lineage.shape[0] / self.n_gens)

        nodes = self.ID[-self.n_inds:]
        left = 0
        right = n_loci-1
        haps = set([Haplotype(n, left, right, [n]) for n in nodes])

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
        num_haps = len(self.haps)
        new_haps = []
        old_haps = []
        for hap in self.active_haps():
            node = hap.node
            ## Format is [Offspring, Parent, StartChrom, rec1, rec2, ...]
            recs = self.recs[node][3:]

            ## Only split if both sides of breakpoint are in Haplotype
            recs = np.array([r for r in recs if hap.left <= r < hap.right])
            assert len(recs) == len(set(recs))

            ## Replace old haplotype with new split
            if len(recs) > 0:
                new_haps.extend(hap.split(recs))
                old_haps.append(hap)

        for h in old_haps:
            assert h in self.haps
            self.haps.discard(h)

        self.haps.update(new_haps)

        ## We should have omre haplotypes after recombination
        assert len(self.haps) >= num_haps

    def climb(self):
        """ Climb all haplotypes to their parent node """
        for hap in self.active_haps():
            parent_ix = hap.left
            ##TODO: Change to use recs instead of lineage - more robust +t1
            new_node = self.lineage[hap.node][parent_ix]
            print(hap.node, "climbs to", new_node)

            ## Negative nodes indicate the top of the lineage, which we
            ## store as an inactive node unless the haplotype was just created
            if new_node < 0:
                ## Newly created haplotypes start as their own child
                if hap.node not in hap.children:
                    self.haps.add(Haplotype(hap.node, hap.left, hap.right,
                                  hap.children, active=False))
            else:
                self.haps.add(hap.climb(new_node))
            self.haps.discard(hap)

        if reached_top is False:
            assert len(self.haps) == num_haps


    def coalesce(self):
        """
        Coalesces haplotypes that share loci within a common ancestor
        """
        for node, haps in self.collect_active_haps().items():

            if len(haps) > 1:
                regions = [(h.left, h.right) for h in haps]
                new_regions = region_overlap(regions)

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
                    left = new_region[0]
                    right = new_region[1]
                    coalesced_hap = Haplotype(node, left, right, children,
                                              active=active)
                    self.haps.add(coalesced_hap)

                    ## If coalescence happened, create a new haplotype to
                    ## continue climbing
                    if active is False:
                        self.haps.add(Haplotype(node, left, right, [node]))

                ## Remove old haplotypes
                for h in haps:
                    self.haps.discard(h)


    def trace(self):
        """ Traces coalescent events through the population lineage """
        while len(self.active_haps()) > 0:
            self.recombine()
            self.climb()
            self.coalesce()


    def tree_sequence(self):
        """
        Returns the coalescent history of the population as an msprime
        TreeSequence object
        """
        nodes = msprime.NodeTable()
        edgesets = msprime.EdgesetTable()

        ## Add rows to msprime NodeTable
        for ID, time in zip(self.ID, self.times):
            is_sample = np.uint32(time == 0)
            nodes.add_row(time=time, population=0, flags=is_sample)

        ## Store edgesets data in structured array to simplify sorting
        dtypes = [('left', np.uint32), ('right', np.uint32),
                  ('parent', np.int32), ('time', np.float)]
        edge_array = np.zeros(len(self.haps))
        edge_array = np.array(edge_array, dtype=dtypes)

        ## Build arrays for constructing edgesets table
        haps = np.array(list(self.haps))
        for i, hap in enumerate(haps):
            edge_array[i]['left'] = hap.left
            edge_array[i]['right'] = hap.right
            edge_array[i]['parent'] = hap.node
            edge_array[i]['time'] = self.times[hap.node]

        ## Ensure arrays are sorted by ascending parent time and increasing
        ## left segment value
        order = np.argsort(edge_array, order=['time', 'left'])
        ordered_edgesets = edge_array[order]

        children = []
        children_length = []
        for i in order:
            children.extend(haps[i].children)
            children_length.append(len(haps[i].children))

        ## Construct msprime edgesets table
        edgesets.set_columns(left=ordered_edgesets['left'],
                 right=ordered_edgesets['right'],
                 parent=ordered_edgesets['parent'],
                 children=np.array(children).astype(np.int32),
                 children_length=np.array(children_length).astype(np.uint32))

        ts = msprime.load_tables(nodes=nodes, edgesets=edgesets)

        return ts

