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

## Each item in the list will be passed once to the tests
params = [args] * 30


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

    ID = FSim.get_idx(FSim.ID).ravel()
    lineage = FSim.get_idx(FSim.lineage)
    recs = FSim.recs

    yield {'TreeSequence': ts, 'haplotypes': simuPOP_haps,
            'positions': positions, 'simuPOP_pop': pop,
            'args': args, 'ID': ID, 'lineage': lineage,
            'recs': recs}


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


def generate_pop():
    ID = np.arange(9)
    lineage = np.array([[-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [ 0,  1,  0,  1,  1],
                        [ 1,  1,  1,  1,  2],
                        [ 1,  1,  2,  2,  2],
                        [ 3,  3,  4,  3,  3],
                        [ 5,  4,  4,  4,  5],
                        [ 5,  5,  5,  4,  4]])

    ## Currently the first three columns are ignored - lineage is read from
    ## separate array
    recs = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 3],
                     [0, 0, 0, 1],
                     [0, 0, 0, 1, 2],
                     [0, 0, 0, 0, 3],
                     [0, 0, 0, 2]])
    n_gens = 3

    P = wf_trace.Population(ID, lineage, recs, n_gens)
    return P


def check_gen(Population):
    for hap in Population.haps:
        assert len(hap.loci) >= 1
        assert len(hap.children) >= 1
        assert hap.node >= 0
        if len(hap.loci) > 1:
            assert set(np.diff(hap.loci)) == set([1])


def test_pop(source_pops):
    ID = source_pops['ID']
    lineage = source_pops['lineage']
    recs = source_pops['recs']
    n_gens = source_pops['args'].t_admix

    ## Initialize population
    P = wf_trace.Population(ID, lineage, recs, n_gens)

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
        
