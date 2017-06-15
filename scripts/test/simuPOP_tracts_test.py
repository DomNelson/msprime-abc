import sys, os
import numpy as np
import pytest
sys.path.append(os.path.abspath('../'))
import argparse
import simuPOP_tracts



## Each item in the list will be passed once to the tests
props = [[0.9, 0.1],
         [0.2, 0.3, 0.5]]
params = []
for prop in props:
    args = argparse.Namespace(
            n_inds=100,
            n_sites=100,
            props=prop,
            n_gens=10)

    params.append(args)


@pytest.fixture(scope='module', params=params)
def pop_tracts(request):
    args = request.param

    ## Generate pop and evolve it
    pop = simuPOP_tracts.initialize_pop(args.n_inds, args.n_sites, args.props)
    evolved_pop = simuPOP_tracts.evolve_pop(pop, args.n_gens)

    ## Get haplotypes from ancestral and current populations
    haplotypes = simuPOP_tracts.pop_haplotypes(pop)
    evolved_haplotypes = simuPOP_tracts.pop_haplotypes(evolved_pop)

    ## Get binned tract lengths
    tract_lengths = simuPOP_tracts.pop_tracts(pop)
    evolved_tract_lengths = simuPOP_tracts.pop_tracts(evolved_pop)

    yield {'args': args, 'pop': pop, 'evolved_pop': evolved_pop,
            'haplotypes': haplotypes, 'evolved_haplotypes': evolved_haplotypes,
            'tract_lengths': tract_lengths,
            'evolved_tract_lengths': evolved_tract_lengths}


def test_pop(pop_tracts):
    args = pop_tracts['args']
    pop = pop_tracts['pop']
    evolved_pop = pop_tracts['evolved_pop']
    haplotypes = pop_tracts['haplotypes']
    evolved_haplotypes = pop_tracts['evolved_haplotypes']
    tract_lengths = pop_tracts['tract_lengths']
    evolved_tract_lengths = pop_tracts['evolved_tract_lengths']

    ## Make sure we've simulated a diploid population
    assert len(haplotypes) == args.n_inds * pop.ploidy()
    assert len(evolved_haplotypes) == args.n_inds * pop.ploidy()

    ##TODO Check for evenly-spaced positions +t1

    ## Make sure each individual has the right number of sites
    for ind, new_ind in zip(pop.individuals(), evolved_pop.individuals()):
        assert len(ind.genotype()) == 2 * args.n_sites
        assert len(new_ind.genotype()) == 2 * args.n_sites

        ## Make sure we don't have any new mutations
        possible_ancestries = range(pop.dvars().n_ancestries)
        assert set(ind.genotype()).issubset(possible_ancestries)
        assert set(new_ind.genotype()).issubset(possible_ancestries)
