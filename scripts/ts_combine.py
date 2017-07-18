import itertools
import numpy as np
import attr
import msprime


def shift_connecting_edge_array(connecting_edge_array, node_shift):
    """
    Updates parents of edges in connecting edgesets, which come from the
    top tree sequence and have shifted from addition of bottom nodes
    """
    ## Update node indices in connecting edgesets
    shift_parent = [p + node_shift for p in connecting_edge_array['parent']]
    connecting_edge_array['parent'] = shift_parent

    return connecting_edge_array


def get_uncoalesced(ts, great_anc):
    """
    Returns children and intervals of descendants of artificially
    coalesced node, denoted great_anc
    """
    for t in ts.trees():
        if t.get_root() == great_anc:
            yield t.children(t.get_root()), t.interval


def combine_nodes(top_nodes, bottom_nodes):
    """
    Combines all nodes provided into a single msprime
    node table
    """
    nodes_table = msprime.NodeTable()

    for node in itertools.chain(bottom_nodes, top_nodes):
        nodes_table.add_row(**vars(node))
        
    return nodes_table


def shuffle_leaves(ts):
    """
    Returns shuffled samples top tree, so they can be
    connected randomly to the bottom tree sequence
    """
    leaves = ts.samples()
    np.random.shuffle(leaves)

    return leaves


def connecting_edge_array(ts_top, ts_bottom, great_anc):
    """
    Returns edgesets connecting the two trees, where connections to
    great_anc are cut in ts_bottom, and attached to random nodes
    in ts_top
    """
    dtypes = [('left', np.float), ('right', np.float),
        ('parent', np.int32), ('children', tuple)]

    leaves = shuffle_leaves(ts_top)
    uncoal = list(get_uncoalesced(ts_bottom, great_anc))
    
    ## Store edge attributes in structured array
    edge_array = np.zeros(len(uncoal), dtype=dtypes)

    for i, (leaf, (children, interval)) in enumerate(zip(leaves, uncoal)):
        edge_array[i] = (interval[0], interval[1], leaf, children)

    return edge_array


def edge_array_to_table_records(edge_array):
    """
    Converts a structured array with children a a tuple to a structured
    array which stores children length and flat list of children, and can
    be imported directly into an msprime EdgesetTable
    """
    dtypes = [('left', np.float), ('right', np.float),
        ('parent', np.int32), ('children_length', np.uint32)]
    edge_records = np.zeros(edge_array.shape[0], dtype=dtypes)

    left = edge_array['left']
    right = edge_array['right']
    parent = edge_array['parent']
    children = edge_array['children']

    all_children = []
    for i, (l, r, p, c) in enumerate(zip(left, right, parent, children)):
        edge_records[i] = (l, r, p, len(c))
        all_children.extend(c)

    return edge_records, all_children


def edgesets_to_edge_records(edgesets):
    """
    Returns edgeset attributes as a numpy structured array
    """
    dtypes = [('left', np.float), ('right', np.float),
        ('parent', np.int32), ('children_length', np.uint32)]
    edgesets = list(edgesets)

    edge_records = np.zeros(len(edgesets), dtype=dtypes)
    children = []
    for i, e in enumerate(edgesets):
        edge_records[i] = (e.left, e.right, e.parent, len(e.children))
        children.extend(e.children)

    return edge_records, children


def edgesets_table_from_records(*edge_records_list):
    """
    Combines structured arrays of edge attributes into single msprime
    EdgesetTable
    """
    records, children = zip(*edge_records_list)
    edge_records = np.hstack(records)
    all_children = np.hstack(children).astype(np.int32)

    edgesets_table = msprime.EdgesetTable()
    edgesets_table.set_columns(left=edge_records['left'],
             right=edge_records['right'],
             parent=edge_records['parent'],
             children=all_children,
             children_length=edge_records['children_length'])
    
    return edgesets_table


@attr.s
class TSCombine(object):
    ts_bottom = attr.ib()
    ts_top = attr.ib()
    time_shift = attr.ib(default=None)
    
    
    def __attrs_post_init__(self):
        self.top_nodes = list(self.ts_top.nodes())
        self.bottom_nodes = list(self.ts_bottom.nodes())
        self.top_edgesets = list(self.ts_top.edgesets())
        self.bottom_edgesets = list(self.ts_bottom.edgesets())
        
        ## Node used to coalesce lineages remaining after wf simulation
        self.great_anc = [i for i, n in enumerate(self.ts_bottom.nodes())
                                if n.name == 'great_anc_node'][0]

        if self.time_shift is None:
            self.time_shift = self.get_shift()
            

    def get_shift(self):
        """
        Gets time difference between top and bottom matching nodes
        """
        ##TODO: Should be determined by time of great_anc, if present +t1
        top_times = [self.top_nodes[n].time for n in self.match_top]
        bottom_times = [self.bottom_nodes[n].time for n in self.match_bottom]

        top_time = top_times[0]
        bottom_time = bottom_times[0]
        time_shift = bottom_time - top_time
        
        return time_shift


    def shift_top_edgesets(self):
        """
        Updates nodes in top edgesets to compensate for bottom nodes being
        prepended to top nodes
        """
        ## Update node indices in edgesets of top tree sequence
        node_shift = len(self.bottom_nodes)
        for e in self.top_edgesets:
            e.parent += node_shift
            e.children = tuple(c + node_shift for c in e.children)


    def shift_top_node_times(self):
        """
        Update top node times to compensate for new zero time in bottom
        tree sequence
        """
        for n in self.top_nodes:
            n.time += self.time_shift
            if len(n.name) == 0:
                n.name = 'top'


    def update_samples(self, nodes):
        """
        Removes sample flag from nodes which gained children
        """
        for n_ix in nodes:
            assert self.top_nodes[n_ix].flags == 1
            self.top_nodes[n_ix].flags = 0


    def combine(self):
        """
        Returns a tree sequence formed by overlapping the specified nodes
        """
        ## Get edges leading to root of the bottom tree, which will be
        ## attached to the leaves of the top tree
        ca = connecting_edge_array(self.ts_top, self.ts_bottom, self.great_anc)

        ## Remove sample status from top leaves which connect to bottom tree
        ## sequence 
        self.update_samples(ca['parent'])

        ## Align node indices to match new combined list
        node_shift = len(self.bottom_nodes)
        ca = shift_connecting_edge_array(ca, node_shift)
        ce = edge_array_to_table_records(ca)
        self.shift_top_node_times()
        self.shift_top_edgesets()

        ## Combine bottom, connecting, and top edgesets into a table
        te = edgesets_to_edge_records(self.top_edgesets)
        be = edgesets_to_edge_records(self.bottom_edgesets)
        edgesets_table = edgesets_table_from_records(be, ce, te)

        ## Combine nodes from each tree sequence
        nodes_table = combine_nodes(self.top_nodes, self.bottom_nodes)

        combined_ts = msprime.load_tables(nodes=nodes_table,
                                          edgesets=edgesets_table)
        
        return combined_ts
    
    
def combine(top_ts, bottom_ts, node_matches, time_shift=None):
    """
    Returns a tree sequence formed by overlapping the specified nodes
    """
    TSC = TSCombine(top_ts, bottom_ts, node_matches, time_shift)
    TSC.align()
    
    return TSC.combine()
