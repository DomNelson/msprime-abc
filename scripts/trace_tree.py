import numpy as np
import tables
import copy
import msprime
from profilehooks import timecall
import attr
from collections import defaultdict
import _msprime
from _msprime import MutationGenerator, RandomGenerator


def get_chrom(idx, start_chrom, breakpoints):
    """
    Returns the chromosome in which idx falls, given the recombinations
    provided, as either -1 or 1
    """
    if len(breakpoints) == 0:
        return start_chrom

    for i, b in enumerate(breakpoints):
        if b >= idx:
            return (i + start_chrom) % 2

    return (i + start_chrom - 1) % 2


def bool_to_signed(bool_val):
    """
    Converts 0/1 to 1/-1
    """
    return np.sign(-1 * bool_val + 0.5).astype(int)


def signed_to_bool(signed_val):
    """
    Converts 1/-1 to 0/1
    """
    return int(-1 * (signed_val - 1) / 2)


def region_overlap(regions):
    """
    Returns a dict with format {(*old_regions_ix,): new_region} where
    each new_region has a unique coalescence pattern
    """
    points = sorted(set().union(*regions))

    for i in range(1, len(points)):
        old_haps = [j for j, r in enumerate(regions)
                        if points[i-1] >= r[0] and r[1] >= points[i]]
        
        ## Skip regions between disjoint segments
        if len(old_haps) > 0:
            yield old_haps, points[i-1], points[i]


def collect_children(haps):
    """ Returns a tuple containing all children of the provided haplotypes """
    children = []
    for hap in haps:
        children.extend(hap.children)

    assert len(children) == len(set(children))

    return tuple(sorted(children))


def coalesce_haps(common_anc_haps):
    """
    Coalesces haplotypes that share loci within a common ancestor
    """
    for node, haps in common_anc_haps:
        assert np.array([(h.node == node) for h in haps]).all()
        regions = [(h.left, h.right) for h in haps]

        for old_hap_idx, left, right in region_overlap(regions):
            ## Collect child nodes of coalesced haplotypes
            old_haps = [haps[i] for i in old_hap_idx]
            children = collect_children(old_haps)
            assert len(children) > 0

            ## If there is only haplotype in this region, then no
            ## coalescence happened in this segment and it stays active
            active = (len(old_haps) == 1)

            ## Create updated haplotype
            time = haps[0].time
            yield Haplotype(node, left, right, children, time, active)

            ## If coalescence happened, create a new haplotype to
            ## continue climbing
            if active is False:
                assert len(children) > 1
                yield Haplotype(node, left, right, [node], time, True)


def recombine_haps(haps, rec_vec):
    """
    Climbs haplotypes to the appropriate parent node, splitting them
    if recombination occurred
    """
    ##TODO Check if these lists are necessary +t3
    num_haps = len(haps)
    num_recs = 0
    new_haps = []
    old_haps = []
    for hap in haps:
        ## Format is [Offspring, Parent, StartChrom, rec1, rec2, ...]
        recs = rec_vec[3:]

        ## Only split if both sides of breakpoint are in Haplotype. Note that
        ## hap.right is not included in the haplotype, and splits are 
        ## defined as happening after the recombination index
        recs = np.array([r for r in recs[3:] if hap.left <= r < hap.right-1])
        num_recs += len(recs)
        assert len(recs) == len(set(recs))

        ## Replace old haplotype with new split
        if len(recs) > 0:
            new_haps.extend(hap.split(recs))
            old_haps.append(hap)

    return old_haps, new_haps

    ## We should have more haplotypes after recombination
    assert len(self.haps) >= num_haps
    # assert len(self.haps) == num_haps + num_recs


@attr.s(frozen=True)
class Haplotype(object):
    node = attr.ib(convert=int)
    left = attr.ib(convert=int)
    right = attr.ib(convert=int)
    children = attr.ib(convert=tuple)
    time = attr.ib(convert=int)
    active = attr.ib(default=True)


    def __attrs_post_init__(self):
        # assert self.node != 0 # - used as label for great anc of whole pop
        assert len(self.children) > 0
        assert self.left <= self.right


    def climb(self, node):
        """ Climbs to the specified node """
        time = self.time + 1
        return Haplotype(node, self.left, self.right, self.children, time)


    def split(self, loci_ix):
        """
        Returns new haplotypes, representing the current haplotype split at
        the given loci
        """
        ## Make sure we don't duplicate points if loci_ix contains self.left
        loci_ix = np.array(loci_ix)
        pts = sorted(set([self.left] + list(loci_ix+1) + [self.right]))
        assert ((loci_ix >= self.left) & (loci_ix < self.right-1)).all()

        for i in range(len(pts)-1):
            left = pts[i]
            right = pts[i+1]
            yield Haplotype(self.node, left, right, self.children, self.time)


@attr.s
class Population(object):
    """
    A population of haplotypes, with methods for retracing coalescent trees
    through a lineage constructed by forward simulations
    """
    coalesce_all = attr.ib(default=True)
    sample_size = attr.ib(default='all')


    def __attrs_post_init__(self):
        self.uncoalesced_haps = []

        ## Node used to coalesce lineages remaining after wf simulation
        self.great_anc_node = 0


    def init_haps(self, node_IDs, L):
        """
        Initializes haplotypes according to ID labels, in last generation,
        with loci numbered sequentially
        """
        if self.sample_size != 'all':
            nodes = np.random.choice(node_IDs, size=self.sample_size)

        left = 0
        right = L
        time = 0

        ## Initialize sample nodes, which do not climb
        self.haps = set([Haplotype(n, left, right, [n], time, active=False)
                    for n in node_IDs])

        ## Initialize haplotypes to climb from samples
        self.haps.update(set([Haplotype(n, left, right, [n], time)
                    for n in node_IDs]))


    def active_haps(self):
        """ Returns current active lineages in the population """
        return [h for h in self.haps if h.active is True]


    def common_anc_haps(self):
        """ Returns active haplotypes which share a common ancestor """
        nodes = defaultdict(list)
        
        for hap in self.active_haps():
            nodes[hap.node].append(hap)

        for node, haps in nodes.items():
            if len(haps) > 1:
                yield node, haps


    def detailed_lineage(self):
        """
        For debugging, shows ID, lineage, and recombinations side-by-side
        """
        return np.hstack([self.ID[self.n_inds:].reshape(-1, 1),
                          np.array(self.recs).reshape(-1, 1)])


    def climb(self, rec_vecs):
        """ Climb all haplotypes to their parent node """
        for hap, rec_vec in zip(self.active_haps(), rec_vecs):
            offspring, parent, start_chrom, *breakpoints = rec_vec
            assert offspring == np.abs(hap.node)

            ## Find which parental chromosome the haplotype inherits from,
            ## given that chrom is +-1 and chrom labels are signed
            chrom = get_chrom(hap.left, start_chrom, breakpoints)
            signed_chrom = bool_to_signed(chrom)
            new_node = parent * signed_chrom

            if new_node in self.founders:
                ## Store founders as an inactive node
                founder_hap = Haplotype(new_node, hap.left, hap.right,
                              hap.children, hap.time, active=False)
                self.haps.add(founder_hap)
                self.uncoalesced_haps.append(founder_hap)
                self.haps.discard(hap)
                continue

            ## Climb to new node and discard old haplotype
            self.haps.add(hap.climb(new_node))
            self.haps.discard(hap)


    def coalesce(self):
        """
        Resolves coalescences within common ancestor events and updates
        haplotypes 
        """
        ## Collect and coalesce haplotypes
        common_anc_haps = list(self.common_anc_haps())
        new_haps = coalesce_haps(common_anc_haps)

        ## Replace old haplotypes with new coalesced ones
        for node, haps in common_anc_haps:
            for h in haps:
                self.haps.discard(h)

        self.haps.update(new_haps)


    def recombine(self, rec_vec):
        """
        Returns new haplotypes which have been created through recombination
        """
        haps = self.active_haps()
        old_haps, new_haps = recombine_haps(haps, rec_vec)

        for h in old_haps:
            assert h in self.haps
            self.haps.discard(h)

        self.haps.update(new_haps)


    def trace(self):
        """ Traces coalescent events through the population lineage """
        ##NOTE: This is where we can decide to stop as soon as all lineages
        ## have coalesced +n1
        while len(self.active_haps()) > 0:
            print(len(self.active_haps()))
            self.recombine()
            self.climb()
            self.coalesce()

        if self.coalesce_all is True:
            print("Coalescing all remaining haps")
            ## If set, coalesce all remaining haplotypes in a new node
            for hap in self.uncoalesced_haps:
                self.haps.add(hap.climb(self.great_anc_node))
                self.haps.discard(hap)

            self.coalesce()


@attr.s
class TreeBuilder(object):
    haps = attr.ib(convert=list)
    positions = attr.ib(convert=list)
    pop_dict = attr.ib(default=None)


    def __attrs_post_init__(self):
        ## Default population is -1
        if self.pop_dict is not None:
            self.pop_dict = defaultdict(lambda: -1, self.pop_dict)
        else:
            self.pop_dict = defaultdict(lambda: -1)

        ## Node used to coalesce lineages remaining after wf simulation
        self.great_anc_node = 0

        self.nodes = msprime.NodeTable()
        self.edgesets = msprime.EdgesetTable()
        self.haps_idx = {}
        self.idx_haps = {}
        self.ts = self.tree_sequence()


    def add_nodes(self):
        """
        Builds list of nodes from provided haplotypes, storing relationship
        between hap ID and node list index
        """
        i = 0
        for hap in self.haps:
            ## Store one node per individual, for all segments
            if hap.node not in self.haps_idx:
                is_sample = np.uint32(hap.time == 0)

                if hap.node == self.great_anc_node:
                    name = 'great_anc_node'
                else:
                    name = ''

                self.nodes.add_row(time=hap.time,
                                   population=self.pop_dict[np.abs(hap.node)],
                                   flags=is_sample, name=name)
                self.haps_idx[hap.node] = i
                self.idx_haps[i] = hap.node
                i += 1


    def hap_array(self):
        """
        Returns haplotype data in a structured array to facilitate contructing
        msprime edgesets
        """
        edge_records = []
        for hap in self.haps:
            children = sorted([self.haps_idx[c] for c in hap.children])

            ## Exclude samples and uncoalesced lineages
            if hap.time > 0 and len(children) > 1:
                assert hap.active is False
                assert self.nodes.flags[self.haps_idx[hap.node]] == 0

                edge_records.append((hap.left, hap.right,
                                    self.haps_idx[hap.node],
                                    hap.time, children, len(children)))

        ## Store edgesets data in structured array to simplify sorting
        dtypes = [('left', np.float), ('right', np.float),
                  ('parent', np.int32), ('time', np.float),
                  ('children', tuple), ('children_length', np.uint32)]

        ## Ensure arrays are sorted by ascending parent time and increasing
        ## left segment value
        array = np.core.records.fromrecords(edge_records, dtype=dtypes)
        ordered_array = np.sort(array, order=['time', 'left'])

        return ordered_array


    def add_edges(self):
        """
        Adds edges to msprime edgeset object from data in haplotype list
        """
        edge_array = self.hap_array()
        children = [c for tup in edge_array['children'] for c in tup]

        ## Construct msprime edgesets table
        self.edgesets.set_columns(left=edge_array['left'],
                 right=edge_array['right'],
                 parent=edge_array['parent'],
                 children=children,
                 children_length=edge_array['children_length'])


    def sorted_records(self):
        """
        For debugging, returns coalescence records in the order in which they
        are stored in self.edgesets
        """
        children_idx = 0
        for i in range(self.edgesets.num_rows):
            parent = self.edgesets.parent[i]
            left = self.edgesets.left[i]
            right = self.edgesets.right[i]

            l = self.edgesets.children_length[i]
            children = self.edgesets.children[children_idx:children_idx+l]
            children_idx += l

            print(parent, left, right, children)


    def tree_sequence(self):
        """
        Returns the coalescent history of the population as an msprime
        TreeSequence object
        """
        self.add_nodes()
        self.add_edges()

        ts = msprime.load_tables(nodes=self.nodes, edgesets=self.edgesets)
        ts.simplify()
        samples = [h for h in self.haps if h.time == 0]
        assert len(list(ts.get_samples())) == len(samples)

        return ts


    def genotypes(self, nodes, h5file):
        """
        Returns the full genotype associated with the provided node
        """
        with tables.open_file(h5file, 'r') as f:
            ind_IDs = f.root.inds[:]
            uID_idx = dict([(ID, i) for i, ID in enumerate(ind_IDs)])

            genotypes = {}
            for node in nodes:
                ID = self.idx_haps[node]
                uID = np.abs(ID).astype(int)

                try:
                    file_idx = uID_idx[uID]
                except KeyError:
                    print("No genotype for", node)
                    continue

                chrom = signed_to_bool(np.sign(ID))
                genotypes[node] = f.root.haps[file_idx][chrom]
                assert f.root.inds[file_idx] == uID

        return genotypes


def mutate_ts(ts, mu, seed=None):
    """
    Throws down mutations at rate mu on the provided tree sequence
    """
    if seed is None:
        seed = np.random.randint(1, 2**32)

    rng = RandomGenerator(seed)
    m = MutationGenerator(rng, mu)

    ##TODO: May be a cleaner way of handling these tables +t3
    ts_tables = ts.dump_tables()
    node_table = ts_tables.nodes
    edgeset_table = ts_tables.edgesets
    migrations_table = ts_tables.migrations
    mutation_table = ts_tables.mutations
    mutation_type_table = ts_tables.sites

    ## Generate mutations on the tree sequence
    m.generate(node_table, edgeset_table, mutation_type_table,
                mutation_table)

    ## Create new tree sequence containing the mutations
    ts = ts.load_tables(nodes=node_table, edgesets=edgeset_table,
                    mutations=mutation_table, sites=mutation_type_table,
                    migrations=migrations_table)

    return ts
