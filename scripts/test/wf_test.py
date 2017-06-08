import sys, os
import numpy as np
import pytest
sys.path.append(os.path.abspath('../'))
import argparse
import msprime_wf


args = argparse.Namespace(
        Na=10,
        t_admix=10,
        t_div=1000,
        admixed_prop=0.5,
        rho=1e-8,
        mu=1e-8,
        length=1e6
        )

## Each item in the list will be passed once to the tests
params = [args]


@pytest.fixture(scope='module', params=params)
def source_pops(request):
    args = request.param
    ts = msprime_wf.generate_source_pops(args)

    simuPOP_haps = msprime_wf.msprime_hap_to_simuPOP(ts)
    positions = msprime_wf.msprime_positions(ts)
    pop = msprime_wf.wf_init(simuPOP_haps, positions)

    yield {'TreeSequence': ts, 'haplotypes': simuPOP_haps,
            'positions': positions, 'simuPOP_pop': pop,
            'args': args}


def test_pop(source_pops):
    args = source_pops['args']
    pop = source_pops['simuPOP_pop']
    haplotypes = source_pops['haplotypes']
    positions = source_pops['positions']

    ## Make sure we've simulated a diploid population
    assert len(haplotypes) == 2 * args.Na

    ## Integer positions are suspicious, ie. not randomly generated
    for pos in positions:
        assert int(pos) != pos

    ## Make sure haplotypes and positions align
    n_sites = len(positions)
    for hap in haplotypes:
        assert len(hap) == n_sites

    ## Make sure simuPOP population has proper haplotypes and positions
    for ind in pop.individuals():
        assert len(ind.genotype()) == 2 * n_sites
        assert np.array_equal(np.array(pop.lociPos()), positions)
