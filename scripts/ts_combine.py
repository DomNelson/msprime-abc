import itertools
import numpy as np
import attr
from collections import defaultdict
import msprime


def get_parent_edgeset(edgesets, nodes):
    """
    Returns edgesets which are parental to the nodes provided
    """
    nodes = set(nodes)
    for i, e in enumerate(edgesets):
        children = nodes.intersection(e.children)
        if len(children) > 0:
            for child in children:
                yield child, i


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

        ## If time between tree sequences is not specified, default to
        ## the difference between top leaves and children of bottom root
        ##TODO: Implement +t2
        if self.time_shift is None:
            self.time_shift = self.get_shift()

        ## Track which nodes have been connected so the same individual
        ## is always assigned the same new node. If node hasn't been
        ## connected yet, draws a random leaf
        shuffled_leaves = self.ts_top.samples()
        np.random.shuffle(shuffled_leaves)
        shuffled_leaves = iter(shuffled_leaves)
        random_leaf = lambda: next(shuffled_leaves)
        self.connection_dict = defaultdict(random_leaf)
            

    def get_shift(self):
        """
        Gets time difference between top and bottom matching nodes
        """
        ##TODO: Should be determined by time of great_anc, if present +t1
        top_times = [self.top_nodes[n].time for n in self.match_top]
        bottom_times = [self.bottom_nodes[n].time for n in self.match_bottom]

        top_time = top_times[0]
        bottom_time = np.max(bottom_times)
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


    def get_connections(self):
        """
        Returns edgesets connecting the two trees, where connections to
        great_anc are cut in ts_bottom, and attached to random nodes
        in ts_top
        """
        uncoal = list(get_uncoalesced(self.ts_bottom, self.great_anc))
        
        new_child_dict = defaultdict(set)

        for i, (children, interval) in enumerate(uncoal):
            for child in children:
                leaf = self.connection_dict[child]
                new_child_dict[leaf].add(child)

        return dict(new_child_dict)


    def make_connections(self, new_child_dict, node_shift):
        shifted_nodes = [n + node_shift for n in new_child_dict.keys()]
        leaf_parents = get_parent_edgeset(self.top_edgesets, shifted_nodes)
        parent_dict = dict(leaf_parents)

        for leaf, children in new_child_dict.items():
            assert len(children) == 1
            shifted_leaf = leaf + node_shift
            edge_index = parent_dict[shifted_leaf]
            edgeset = self.top_edgesets[edge_index]

            print(children)
            print("Connecting to", edgeset)
            new_children = list(self.top_edgesets[edge_index].children)
            new_children = tuple(sorted(new_children + list(children)))

            self.top_edgesets[edge_index].children = new_children


    def combine(self):
        """
        Returns a tree sequence formed by overlapping the specified nodes
        """
        ## Align node indices to match new combined list
        node_shift = len(self.bottom_nodes)
        self.shift_top_node_times()
        self.shift_top_edgesets()

        ## Connect uncoalesced lineages from the bottom tree to parents of
        ## leaves in the top tree
        new_child_dict = self.get_connections()
        self.make_connections(new_child_dict, node_shift)

        ## Remove sample status from top leaves which connect to bottom tree
        ## sequence 
        self.update_samples(new_child_dict.keys())

        ## Combine nodes from each tree sequence
        nodes_table = combine_nodes(self.top_nodes, self.bottom_nodes)

        ## Combine bottom, connecting, and top edgesets into a table
        te = edgesets_to_edge_records(self.top_edgesets)
        be = edgesets_to_edge_records(self.bottom_edgesets)
        edgesets_table = edgesets_table_from_records(be, te)

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
