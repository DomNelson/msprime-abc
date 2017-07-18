import sys, os
import numpy as np
import pytest
sys.path.append(os.path.abspath('../'))
import argparse
import copy
import trace_tree
import pop_models
import forward_sim as fsim


args = argparse.Namespace(
        n_inds=100,
        n_gens=10,
        rho=1e-8,
        L=1e8,
        n_loci=100,
        h5_out='gen.h5',
        output='genotypes.txt',
        ploidy=2,
        mu=1e-9,
        t_div=100,
        grid_width=2,
        mig_prob=0.1,
        Ne=1000
        )

args2 = argparse.Namespace(
        n_inds=30,
        n_gens=30,
        rho=1e-8,
        L=1e8,
        n_loci=20,
        h5_out='gen.h5',
        output='genotypes.txt',
        ploidy=2,
        mu=1e-9,
        t_div=100,
        grid_width=1,
        mig_prob=0.1,
        Ne=1000
        )

## Each item in the list will be passed once to the tests
params = [args] * 5 + [args2] * 10


@pytest.fixture(scope='module', params=params)
def source_pop_init(request):
    args = request.param

    ## Generate forward simulations which track lineage
    msp_pop = pop_models.grid_ts(N=args.n_inds*args.ploidy,
                    rho=args.rho, L=args.L, mu=args.mu, t_div=args.t_div,
                    Ne=args.Ne, mig_prob=args.mig_prob,
                    grid_width=args.grid_width)

    init_pop = pop_models.msp_to_simuPOP(msp_pop)
    FSim_init = fsim.ForwardSim(args.n_gens, init_pop, output=args.output)

    yield {'args': args, 'FSim_init': FSim_init, 'ts_init': msp_pop.ts}


@pytest.fixture(scope='module', params=params)
def source_pops(request):
    args = request.param

    ## Generate forward simulations which track lineage
    msp_pop = pop_models.grid_ts(N=args.n_inds*args.ploidy,
                    rho=args.rho, L=args.L, mu=args.mu, t_div=args.t_div,
                    Ne=args.Ne, mig_prob=args.mig_prob,
                    grid_width=args.grid_width)

    init_pop = pop_models.msp_to_simuPOP(msp_pop)

    ##TODO: Implement mutations in forward sims +t1
    FSim = fsim.ForwardSim(args.n_gens, init_pop, output=args.output)
    FSim.evolve()

    ID = FSim.ID.ravel()
    recs = FSim.recs
    simuPOP_haps, pops = pop_models.msprime_hap_to_simuPOP(msp_pop.ts)
    positions = pop_models.msprime_positions(msp_pop.ts)

    ## Write genotypes to file
    FSim.write_haplotypes(args.h5_out)

    yield {'haplotypes': simuPOP_haps, 'positions': positions,
           'FSim': FSim, 'args': args, 'ID': ID, 'recs': recs}


def test_simuPOP(source_pops):
    args = source_pops['args']
    pop = source_pops['FSim'].pop
    haplotypes = source_pops['haplotypes']
    positions = source_pops['positions']

    ## Make sure we've simulated the right number of individuals, remembering
    ## that homologous chromosomes are concatenated together
    assert len(haplotypes) == args.n_inds

    ## Integer positions are suspicious, ie. not randomly generated
    for pos in positions:
        assert int(pos) != pos

    ##TODO Sanity check for recombination rate +t1

    ## Make sure haplotypes and positions align
    n_sites = len(positions)
    for hap in haplotypes:
        ## Each haplotype contains 'ploidy' concatenated chromosomes
        assert len(hap) == args.ploidy * n_sites

    ## Make sure simuPOP population has proper haplotypes and positions
    for ind in pop.individuals():
        assert len(ind.genotype()) == 2 * n_sites
        assert np.array_equal(np.array(pop.lociPos()), positions)

        ## Make sure we don't have any new mutations
        assert set(ind.genotype()) == set([0, 1])


def test_haps():
    hap1 = trace_tree.Haplotype(node=1, left=0, right=3, children=(1,),
                time=0, active=True)
    hap2 = trace_tree.Haplotype(node=2, left=0, right=3, children=(1,),
                time=0, active=True)

    rec_dict = {1: [1, 10, 0], 2: [2, 20, 1, 0, 1]}

    old_haps, new_haps = trace_tree.recombine_haps([hap1, hap2], rec_dict)
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
        chroms.append(trace_tree.get_chrom(hap.left, start_chrom, breakpoints))

    assert 0 not in np.diff(chroms)

    ## Test coalescence
    common_anc_haps = [(10, [
            trace_tree.Haplotype(node=10, left=0, right=2, children=(1,), time=0),
            trace_tree.Haplotype(node=10, left=0, right=1, children=(2,), time=0)
            ])]

    coalesced = list(trace_tree.coalesce_haps(common_anc_haps))
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
    n_gens = source_pops['args'].n_gens
    n_loci = source_pops['args'].n_loci

    ## Initialize population
    P = trace_tree.Population(ID, recs, n_gens, n_loci)

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
    n_gens = source_pops['args'].n_gens
    n_loci = source_pops['args'].n_loci
    args = source_pops['args']
    FSim = source_pops['FSim']
    positions = source_pops['positions']

    ## Test conversion to msprime TreeSequence
    P = trace_tree.Population(ID, recs, n_gens, n_loci)
    P.trace()

    positions = FSim.pop.lociPos()
    T = trace_tree.TreeBuilder(P.haps, positions)

    ## Check that homologous chromosomes are split properly
    breakpts = [r for rec in recs for r in rec[3:]]
    assert np.max(breakpts) <= len(positions) - 1

    for t in T.ts.trees():
        assert t.num_leaves(t.root) == args.n_inds * 2

    ## Check genotypes along tree lineages
    for t in T.ts.trees():
        left, right = list(map(int, t.interval))
        genotypes = T.genotypes(t.nodes(), args.h5_out)

        for r in t.children(t.root):
            g = set().union([tuple(genotypes[l][left:right])
                                for l in t.leaves(r)])

            ## All nodes should share a single genotype along the tree
            assert len(g) == 1


def test_simuPOP_init(source_pop_init):
    """
    Tests that simuPOP subpopulations are initialized from msprime populations
    properly
    """
    simuPOP_init = source_pop_init['FSim_init'].pop
    ts_init = source_pop_init['ts_init']
    args = source_pop_init['args']
    n_loci = len(list(ts_init.sites()))

    ## Check that proper number of individuals have been created
    simuPOP_n = np.sum([1 for ind in simuPOP_init.individuals()])
    ts_n = len(list(ts_init.haplotypes()))
    assert simuPOP_n == args.n_inds
    assert ts_n == args.n_inds * args.ploidy

    ## Check that genotypes are proper length, remembering that msprime
    ## simulates individual haplotypes, and simuPOP concatenates homologous
    ## chromosomes
    ts_hap = next(ts_init.haplotypes())
    sim_ind = next(simuPOP_init.individuals()).genotype()
    assert len(ts_hap) == n_loci
    assert len(sim_ind) == n_loci * 2

    sim_pop_freqs = list(pop_models.simuPOP_pop_freqs(simuPOP_init))
    ts_freqs = list(pop_models.msprime_pop_freqs(ts_init))

    assert len(sim_pop_freqs) == len(ts_freqs)

    for (sim_pop, sim_freq), (ts_pop, ts_freq) in zip(sim_pop_freqs, ts_freqs):
        assert sim_pop == ts_pop
        assert ts_freq.shape[0] == n_loci
        assert sim_freq.shape[0] == n_loci
        assert np.max(sim_freq) > 0
        print(sim_freq[:20], ts_freq[:20])
        assert (sim_freq == ts_freq).all()





