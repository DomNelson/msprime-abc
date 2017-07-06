import sys, os
import numpy as np
import pytest
sys.path.append(os.path.abspath('../'))
import argparse
import wf_trace
import forward_sim as fsim


args = argparse.Namespace(
        Na=100,
        Ne=1000,
        t_admix=10,
        t_div=1000,
        admixed_prop=0.5,
        rho=1e-8,
        mu=1e-8,
        length=1e6,
        forward=False,
        n_loci=100
        )

args2 = argparse.Namespace(
        Na=30,
        Ne=1000,
        t_admix=30,
        t_div=1000,
        admixed_prop=0.5,
        rho=1e-8,
        mu=1e-8,
        length=1e6,
        forward=False,
        n_loci=10
        )

## Each item in the list will be passed once to the tests
params = [args] * 10 + [args2] * 10


@pytest.fixture(scope='module', params=params)
def source_pops(request):
    args = request.param
    ts = fsim.generate_source_pops(args)

    simuPOP_haps = fsim.msprime_hap_to_simuPOP(ts)
    positions = fsim.msprime_positions(ts)
    pop = fsim.wf_init(simuPOP_haps, positions)

    ## Generate forward simulations which track lineage
    FSim = fsim.ForwardSim(args.Na,
                           args.length,
                           args.t_admix,
                           args.n_loci)

    FSim.evolve()

    ID = FSim.ID.ravel()
    recs = FSim.recs

    yield {'TreeSequence': ts, 'haplotypes': simuPOP_haps,
            'positions': positions, 'simuPOP_pop': pop,
            'args': args, 'ID': ID, 'recs': recs}


def test_simuPOP(source_pops):
    args = source_pops['args']
    pop = source_pops['simuPOP_pop']
    haplotypes = source_pops['haplotypes']
    positions = source_pops['positions']

    ## Make sure we've simulated a diploid population
    assert len(haplotypes) == 2 * args.Na

    ## Integer positions are suspicious, ie. not randomly generated
    for pos in positions:
        assert int(pos) != pos

    ##TODO Sanity check for recombination rate +t1

    ## Make sure haplotypes and positions align
    n_sites = len(positions)
    for hap in haplotypes:
        assert len(hap) == n_sites

    ## Make sure simuPOP population has proper haplotypes and positions
    for ind in pop.individuals():
        assert len(ind.genotype()) == 2 * n_sites
        assert np.array_equal(np.array(pop.lociPos()), positions)

        ## Make sure we don't have any new mutations
        assert set(ind.genotype()) == set([0, 1])


def test_haps():
    hap1 = wf_trace.Haplotype(node=1, left=0, right=3, children=(1,),
                time=0, active=True)
    hap2 = wf_trace.Haplotype(node=2, left=0, right=3, children=(1,),
                time=0, active=True)

    rec_dict = {1: [1, 10, 0], 2: [2, 20, 1, 0, 1]}

    old_haps, new_haps = wf_trace.recombine_haps([hap1, hap2], rec_dict)
    assert set(old_haps) == set([hap2])

    ## Test chromosome splitting
    assert len(new_haps) == 3
    assert new_haps[0].left == 0
    assert new_haps[0].right == 1
    assert new_haps[1].left == 1
    assert new_haps[1].right == 2
    assert new_haps[2].left == 2
    assert new_haps[2].right == 3

    ## Test inheritance tracking
    chroms = []
    for hap in new_haps:
        offspring, parent, start_chrom, *breakpoints = rec_dict[hap.node]
        chroms.append(wf_trace.get_chrom(hap.left, start_chrom, breakpoints))

    assert 0 not in np.diff(chroms)

    ## Test coalescence
    common_anc_haps = [(10, [
            wf_trace.Haplotype(node=10, left=0, right=2, children=(1,), time=0),
            wf_trace.Haplotype(node=10, left=0, right=1, children=(2,), time=0)
            ])]

    coalesced = list(wf_trace.coalesce_haps(common_anc_haps))
    active = [h for h in coalesced if h.active is True]
    inactive = [h for h in coalesced if h.active is False] 

    assert len(coalesced) == 3
    assert len(list(active)) == 2
    assert len(list(inactive)) == 1


def check_gen(Population):
    lefts = []
    rights = []
    for hap in Population.haps:
        assert hap.left < hap.right
        assert len(hap.children) >= 1
        assert hap.node != 0
        lefts.append(hap.left)
        rights.append(hap.right)

    assert len(set(lefts).difference(rights)) == 1
    assert len(set(rights).difference(lefts)) == 1


def test_pop(source_pops):
    ID = source_pops['ID']
    recs = source_pops['recs']
    n_gens = source_pops['args'].t_admix
    n_loci = source_pops['args'].n_loci

    ## Initialize population
    P = wf_trace.Population(ID, recs, n_gens, n_loci)

    for i in range(n_gens):
        check_gen(P)
        start_haps = len(P.haps)

        ## Check recombination step
        P.recombine()
        check_gen(P)
        rec_haps = len(P.haps)
        assert rec_haps >= start_haps

        ## Check climb step
        P.climb()
        check_gen(P)
        assert len(P.haps) == rec_haps

        ## Check coalescence step
        P.coalesce()
        check_gen(P)
        coal_haps = len(P.haps)
        ## Condition below is not strictly true, but should be in the
        ## majority of cases. Uncomment to check +n1
        # assert coal_haps <= rec_haps


def test_treesequence(source_pops):
    ID = source_pops['ID']
    recs = source_pops['recs']
    n_gens = source_pops['args'].t_admix
    n_loci = source_pops['args'].n_loci
    pop = source_pops['simuPOP_pop']
    args = source_pops['args']
    ## Test conversion to msprime TreeSequence

    P = wf_trace.Population(ID, recs, n_gens, n_loci)
    P.trace()

    positions = pop.lociPos()
    T = wf_trace.WFTree(P.haps, positions)
    ts = T.tree_sequence()

    trees = list(ts.trees())

    for t in trees:
        assert t.num_leaves(t.root) == args.Na * 2

        
