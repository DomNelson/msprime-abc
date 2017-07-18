import argparse
import sys, os
sys.path.append('../')
from collections import defaultdict

import wf_tree
import ts_combine


def collect_lineages(haps, haps_idx_dict):
    """
    Returns a dict of uncoalesced lineaged grouped by founding individual
    """
    founder_dict = defaultdict(list)

    for hap in haps:
        children_idx = [haps_idx_dict[c] for c in hap.children]
        founder_dict[hap.node].append((children_idx, (hap.left, hap.right)))

    return founder_dict


args = argparse.Namespace(
        n_inds=20,
        Ne=100,
        n_gens=5,
        rho=1e-8,
        mu=1e-10,
        L=1e7,
        mig_prob=0.25,
        # n_loci=20,
        h5_out='gen.h5',
        MAF=0.1,
        save_genotypes=False,
        track_loci=True
        )

W, initial_pop = wf_tree.main(args)
ts_top = W.ts

W, initial_pop = wf_tree.main(args)
ts_bottom = W.ts

CS = ts_combine.TSCombine(ts_bottom, ts_top, 10)
# CS.align()
cs = CS.combine()

# bottom_roots = [i for i, n in enumerate(ts_bottom.nodes())
#                         if n.time == args.n_gens]
#
# great_anc_node = [i for i, n in enumerate(ts_bottom.nodes())
#                         if n.name == 'great_anc_node'][0]
#
# uncoalesced = [(t.children(t.get_root()), t.interval) for t in ts_bottom.trees()
#                 if t.get_root() == great_anc_node]
#
# # uncoalesced_nodes = [n for i, n in enumerate(ts_bottom.nodes())
# #                         if i in uncoalesced]
#
# lineage_dict = collect_lineages(W.P.uncoalesced_haps, W.T.haps_idx) 
#
# uncoal_dict = defaultdict(lambda: W.T.great_anc_node, W.T.haps_idx)
# bottom_founders = set([uncoal_dict[h.node] for h in W.P.uncoalesced_haps])
