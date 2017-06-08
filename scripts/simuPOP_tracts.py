import sys, os
import numpy as np
import attr
import copy
from profilehooks import profile
from collections import defaultdict, Counter
import simuPOP as sim


def get_tracts(genotype):
    """ Returns tract lengths of each ancestry """
    ##TODO Make sure to split chomosome copies +t1
    genotype = np.array(genotype)
    breakpoints = np.where(np.diff(genotype) != 0)[0]
    tracts = defaultdict(list)

    ## Get the length of the first and last tracts
    tracts[genotype[0]].append(breakpoints[0] + 1)
    tracts[genotype[-1]].append(len(genotype)-breakpoints[-1] - 1)

    ## Tract lengths are the difference between successive breakpoints
    tract_lengths = np.ediff1d(breakpoints)

    for l, b in zip(tract_lengths, breakpoints[1:]):
        tracts[genotype[b]].append(l)

    binned_tracts = {k: Counter(v) for k, v in tracts.items()}

    return binned_tracts


def get_inds_per_ancestry(n_inds, ancestry_props):
    """ Returns number of inds with each ancestry, in list format """
    inds_per_ancestry = (np.array(ancestry_props) * n_inds).astype(int)
    return list(inds_per_ancestry)


def get_genotypes(pop):
    return [ind.genotype() for ind in pop.individuals()]


@attr.s
class PopTracts:
    n_inds = attr.ib()
    n_sites = attr.ib()
    ancestry_props = attr.ib()


    def __attrs_post_init__(self):
        self.n_ancestries = len(self.ancestry_props)

        ## No mutations
        self.mutation_matrix = np.identity(self.n_ancestries).tolist()

        ## Initialize population
        self.initialize_pop()


    def initialize_pop(self):
        """
        Initializes a simuPOP population using the provided parameters
        """
        inds_per_anc = get_inds_per_ancestry(self.n_inds, self.ancestry_props)
        pop = sim.Population(size=inds_per_anc, ploidy=2, loci=self.n_sites)

        ## Set genotypes for each source population as the index of their ancestry
        for i in range(len(self.ancestry_props)):
            sim.initGenotype(pop, haplotypes=[i]*self.n_sites, subPops=[i])

        ## Merge subpopulations
        pop.mergeSubPops(range(self.n_ancestries))

        self.pop = pop


    def evolve_pop(self, ngens, rep=1):
        ## Initialize simulator, without modifying original population
        simu = sim.Simulator(self.pop, stealPops=False, rep=rep)

        ## Evolve with given parameters
        # print("Simulating with mutation matrix", self.mutation_matrix)
        simu.evolve(
            initOps=sim.InitSex(),
            preOps=sim.MatrixMutator(rate=self.mutation_matrix),
            matingScheme=sim.RandomMating(
                    ops=sim.Recombinator(intensity=0.1)),
            gen=ngens
        )

        ##TODO This only returns one replicate +t1
        newpop = simu.extract(0)

        return newpop


    def evolve_genotypes(self, ngens):
        """ Evolves the population ngens, and returns genotypes """
        ##TODO Bug?? Using this instead of calling get_genotypes outside +t2
        ##TODO of class gives different results, namely with mutations +t2
        print("Deprecated - use at own risk")
        pop2 = self.evolve_pop(ngens)

        return get_genotypes(pop2)


def get_pop_tracts(pop):
    """ Returns binned tract lengths for the provided population """
    genotypes = get_genotypes(pop)
    all_tracts = copy.copy(get_tracts(genotypes[0]))

    for g in genotypes[1:]:
        for k, v in get_tracts(g).items():
            all_tracts[k].update(v)

    return all_tracts

P = PopTracts(100, 100, [0.5, 0.5])
p_2 = P.evolve_pop(10)
g_2 = get_genotypes(p_2)
t_2 = get_tracts(g_2[-1])
print(g_2[-1])
print(t_2)
print(get_pop_tracts(p_2))
