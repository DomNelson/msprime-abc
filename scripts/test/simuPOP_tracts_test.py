import sys, os
import numpy as np
import pytest
sys.path.append(os.path.abspath('../'))
import argparse
import simuPOP_tracts


args = argparse.Namespace(
        n_inds=100,
        n_sites=100,
        props=[0.5, 0.2, 0.3],
        n_gens=10)

## Each item in the list will be passed once to the tests
params = [args]


@pytest.fixture(scope='module', params=params)
def pop_tracts(request):
    args = request.param

    ## Generate pop and evolve it
    pop = simuPOP_tracts.initialize_pop(args.n_inds, args.n_sites, args.props)
    evolved_pop = simuPOP_tracts.evolve_pop(pop, args.n_gens)

    ## Get genotypes from ancestral and current populations
    genotypes = simuPOP_tracts.pop_genotypes(pop)
    evolved_genotypes = simuPOP_tracts.pop_genotypes(evolved_pop)

    ## Get binned tract lengths
    tract_lengths = simuPOP_tracts.pop_tracts(pop)
    evolved_tract_lengths = simuPOP_tracts.pop_tracts(evolved_pop)

    yield {'args': args, 'pop': pop, 'evolved_pop': evolved_pop,
            'genotypes': genotypes, 'evolved_genotypes': evolved_genotypes,
            'tract_lengths': tract_lengths,
            'evolved_tract_lengths': evolved_tract_lengths}


def test_pop(pop_tracts):
    args = pop_tracts['args']
    pop = pop_tracts['pop']
    evolved_pop = pop_tracts['evolved_pop']
    genotypes = pop_tracts['genotypes']
    evolved_genotypes = pop_tracts['evolved_genotypes']
    tract_lengths = pop_tracts['tract_lengths']
    evolved_tract_lengths = pop_tracts['evolved_tract_lengths']

    ## Make sure we've simulated a diploid population
    assert len(genotypes) == args.n_inds
    assert len(evolved_genotypes) == args.n_inds

    ## Check for evenly-spaced positions
    ##TODO

    ## Make sure each individual has the right number of sites
    for ind, new_ind in zip(pop.individuals(), evolved_pop.individuals()):
        assert len(ind.genotype()) == 2 * args.n_sites
        assert len(new_ind.genotype()) == 2 * args.n_sites

        ## Make sure we don't have any new mutations
        possible_ancestries = range(pop.dvars().n_ancestries)
        assert set(ind.genotype()).issubset(possible_ancestries)
        assert set(new_ind.genotype()).issubset(possible_ancestries)
