import simuOpt
simuOpt.setOptions(alleleType='lineage', quiet=True)
import tables
from itertools import islice
import attr
import io
import msprime
import simuPOP as sim
import numpy as np
import sys, os

import pop_models
import trace_tree


def ID_increment(start=1):
    i = start
    while True:
        yield i
        i += 1


def draw_breakpoints(L, rho):
    n_recs = np.random.poisson(L * rho)
    recs = np.random.uniform(0, L, size=n_recs)

    return sorted(recs)


@attr.s
class BackwardSim(object):
    n_gens = attr.ib()
    n_inds = attr.ib()
    L = attr.ib()
    rho = attr.ib()


    def __attrs_post_init__(self):
        self.IDs = ID_increment(start=1)


    def next_inds(self, n):
        """
        Returns the next n ind IDs in sequence, which defines a generation
        """
        return list(islice(self.IDs, 0, n))
        

    def init_IDs(self):
        """ Returns IDs for initializing population """
        self.init_IDs_list = self.next_inds(self.n_inds)
        return self.init_IDs_list


    def draw_recomb_vals(self):
        """
        Draws values for a recombination vector with format
            [Offspring, Parent, StartChrom, rec1, rec2, ...]
        and returns as a dict with offspring as key
        """
        prev_gen = self.init_IDs_list

        for i in range(self.n_gens):
            rec_dict = {}

            ##NOTE: Varying population size here +n1
            parent_IDs = self.next_inds(self.n_inds)

            ## Previous generation in initialized in init_IDs
            for ID in prev_gen:
                if ID not in rec_dict:
                    start_chrom = np.random.randint(0, 2)

                    ## Draw parents together, without replacement
                    mother, father = np.random.choice(parent_IDs, size=2,
                                                      replace=False)
                    recs = draw_breakpoints(self.L, self.rho)
                    rec_dict[ID] = [ID, mother, start_chrom] + recs
                    rec_dict[-ID] = [-ID, father, start_chrom] + recs

            prev_gen = parent_IDs

            yield rec_dict


@attr.s
class ForwardSim(object):
    n_gens = attr.ib()
    simuPOP_pop = attr.ib()
    output = attr.ib(default=None)
    track_pops = attr.ib(default=True, convert=bool)
    tracked_loci = attr.ib(default=False)


    def __attrs_post_init__(self):
        self.rho = self.simuPOP_pop.rho
        self.mu = self.simuPOP_pop.mu
        self.pop = self.simuPOP_pop.pop
        self.migmat = self.simuPOP_pop.migmat

        ## Using simuPOP we always have discrete loci
        self.discrete_loci = True

        ## Make sure proper info fields are present to track lineages
        info_fields = ['ind_id', 'chromosome_id', 'allele_id', 'describe',
                        'migrate_to']
        assert set(info_fields).issubset(self.pop.infoFields())

        ##NOTE: Assumes only one chromosome +n1
        assert len(self.pop.numLoci()) == 1
        self.n_loci = self.pop.numLoci()[0]
        assert self.n_loci > 0
        self.L = self.pop.genoSize() / self.pop.ploidy()

        ##NOTE: Assumes constant population size +n1
        n_inds = np.sum(self.pop.subPopSizes()) 
        self.n_inds_per_gen = [n_inds for i in range(self.n_gens)]

        ## Set details of population evolution
        assert (type(self.tracked_loci) is bool or
                                hasattr(self.tracked_loci, '__iter__'))
        self.set_Ops()


    def set_Ops(self):
        ## Create memory buffer to receive file-type output from population
        ## at each generation
        self.ID_IO  = io.StringIO()
        self.recs_IO = io.StringIO()
        self.muts_IO = io.StringIO()

        ## Set operations to be performed during simuPOP population evolution
        self.set_initOps()
        self.set_preOps()
        self.set_matingScheme()
        self.set_postOps()
        self.set_options()


    def set_options(self):
        """
        Updates operators based on initialization args, which act on each
        generation of the forward simulation
        """
        ## Set whether genotypes are output to file or not
        if self.output is not None:
            get_genotype = "str(int(ind.info('ind_id'))) + ','" +\
                            "+ str(ind.genotype()).strip('[]') + '\\n'"

            self.initOps.append(sim.InfoEval(get_genotype, exposeInd='ind',
                          output='>>' + self.output))
            self.postOps.append(sim.InfoEval(get_genotype, exposeInd='ind',
                          output='>>' + self.output))

        ## Set whether deme/subpop info is saved for each individual
        if self.track_pops is True:
            self.subpops = []
            self.initOps.append(sim.PyOperator(self.get_pop_inds))
            self.postOps.append(sim.PyOperator(self.get_pop_inds))

        ## Set whether allele frequencies at specified loci are saved
        if self.tracked_loci is not False:
            self.loci_freqs = []
            self.initOps.append(sim.PyOperator(self.get_freqs))
            self.postOps.append(sim.PyOperator(self.get_freqs))


    def set_initOps(self):
        """
        Sets the operators to be executed at beginning of the forward
        simulation
        """
        get_ID = 'str(int(ind.info("ind_id"))) + "\t"'

        self.initOps = [sim.IdTagger(),
                        sim.InitSex(),
                        sim.InitLineage(mode=sim.FROM_INFO_SIGNED),
                        sim.InfoEval(get_ID, exposeInd='ind',
                                        output=self.ID_IO)]


    def set_preOps(self):
        """
        Sets the operators to be executed before mating in each generation 
        """
        self.preOps = [sim.SNPMutator(u=self.mu, output=self.muts_IO)]

        ## Set migrations if more than one sub-population is present
        if self.pop.numSubPop() > 1:
            self.preOps.append(sim.Migrator(rate=self.migmat))


    def set_matingScheme(self):
        """ Sets the operators to be executing during mating """
        ## Convert rho into an intensity along a simulated chromosome of
        ## length 'n_loci'
        intensity = self.L * self.rho / self.n_loci

        self.matingScheme = sim.RandomMating(ops=[sim.IdTagger(),
                     sim.Recombinator(intensity=intensity,
                                      output=self.recs_IO,
                                      infoFields='ind_id')]) 


    def set_postOps(self):
        """
        Sets operators to be executed after mating in each generation
        """
        get_ID = 'str(int(ind.info("ind_id"))) + "\t"'

        self.postOps = [sim.InfoEval(get_ID, exposeInd='ind',
                                output=self.ID_IO)]


    def evolve(self):
        """
        Traces the lineage of a given number of evenly-spaced loci along a
        single chromosome, in a randomly mating population of size N
        """
        ## Initialize population, making sure that ind IDs start at 1
        sim.IdTagger().reset(1)

        self.pop.evolve(
            initOps=self.initOps,
            preOps=self.preOps,
            matingScheme=self.matingScheme,
            postOps=self.postOps,
            gen=self.n_gens)

        self.parse_sim()


    def get_pop_inds(self, pop):
        """
        For execution during simuPOP population evolution, returns the IDs of
        individuals within each subpopulation
        """
        for sp in range(pop.numSubPop()):
            for ind in pop.individuals(sp):
                self.subpops.extend([ind.ind_id, sp])

        return True


    def get_freqs(self, pop):
        """
        For execution during simuPOP population evolution, returns the allele
        frequency spectrum for each subpopulation
        """
        self.loci_freqs.append(
                list(pop_models.simuPOP_pop_freqs(pop, loci=self.tracked_loci,
                                        ploidy=self.pop.ploidy())))

        return True


    def parse_sim(self):
        raw_ID = list_string_to_np(self.ID_IO.getvalue(), depth=1)
        raw_recs = self.recs_IO.getvalue()
        self.muts = store_mutants(self.muts_IO.getvalue())

        self.recs_IO.close()
        self.ID_IO.close()
        self.muts_IO.close()

        ## Set parsed attributes
        self.subpops = np.array(self.subpops).reshape(-1, 2).astype(int)

        ## Represent chromosomes as signed individual IDs
        signed_ID = np.stack([raw_ID, raw_ID * -1]).T
        self.ID = parse_output(signed_ID, self.n_gens)

        ## Recombinations stay as a list of lists
        self.recs = sort_recombs(raw_recs)


    def init_IDs(self):
        """
        Gets ind IDs for initializing haplotypes for tracing coalescent trees
        """
        n_inds = self.n_inds_per_gen[-1]
        IDs = list(zip(*self.recs[-n_inds:]))[0]

        return IDs


    def draw_recomb_vals(self):
        """
        Draws values for a recombination vector with format
            [Offspring, Parent, StartChrom, rec1, rec2, ...]
        and returns as a dict with offspring as key
        """
        ## Since we're going from the end of the array to the beginning, it's
        ## simpler to explicitly return the first chunk
        n_chroms = self.n_inds_per_gen[-1] * self.simuPOP_pop.pop.ploidy()

        yield dict(zip(self.ID[-n_chroms:].ravel(), self.recs[-n_chroms:]))

        i = -1 * n_chroms
        ##NOTE: Make sure this skipping last gen makes sense +n1
        for n_inds in self.n_inds_per_gen[:-1]:
            n_chroms = n_inds * self.simuPOP_pop.pop.ploidy()
            yield dict(zip(self.ID[i-n_chroms:i].ravel(),
                                                    self.recs[i-n_chroms:i]))
            i = i - n_chroms


    def write_haplotypes(self, h5file):
        """
        Returns an array of genotypes and corresponding chromosome IDs
        """
        filters = tables.Filters(complevel=5, complib='blosc')

        with tables.open_file(h5file, 'w') as h5:
            with open(self.output, 'r') as f:
                line = next(f)
                ind_ID, haplotypes = parse_simuPOP_genotype(line)

		## Create an extendable array in the h5 output file with
		## the same shape as the haplotypes
                h5.create_earray(h5.root, 'haps',
                         atom=tables.IntAtom(shape=(2, haplotypes.shape[2])),
                         shape=(0,), filters=filters)
                h5.create_earray(h5.root, 'inds', atom=tables.IntAtom(),
                         shape=(0,), filters=filters)

                h5.root.haps.append(haplotypes)
                h5.root.inds.append(ind_ID)

                for line in f:
                    ind_ID, haplotypes = parse_simuPOP_genotype(line)
                    h5.root.haps.append(haplotypes)
                    h5.root.inds.append(ind_ID)


def parse_simuPOP_genotype(genotype_string):
    """
    Returns chom_IDs and haplotypes from simuPOP string output
    """
    data = np.fromstring(genotype_string, sep=',')
    ind_ID = data[0].reshape(1)
    haplotypes = data[1:].reshape(1, 2, -1)

    return ind_ID, haplotypes
        
        
def parse_output(output, n_gens, split=False):
    """
    Takes a string representation of simulation output and returns a
    numpy array with individuals having one row per chromosome, and
    the first axis representing time in generations
    """
    output = split_chroms(output)

    if split is True:
        output = split_gens(output, n_gens)

    return output


def list_string_to_np(list_str, depth, astype=int):
    """
    Converts a string representation of a list of lists into a numpy array
    """
    assert depth in [1,2]
    a = list_str.strip().split('\t')

    if depth == 2:
        a = map(lambda x: x.strip('[]').split(','), a)
                
    return np.array(list(a)).astype(astype)


def split_chroms(lineages):
    """
    Takes concatenated chromosomes and splits them into two rows
    """
    i, j = lineages.shape
    assert j % 2 == 0
    
    return lineages.reshape(i*2, int(j/2))


def split_gens(array, n_gens):
    """
    Split a single array output along a new axis representing distinct
    generation
    """
    return np.array(np.split(array, n_gens))


def sort_recombs(rec_str):
    """
    Returns a list of recombinations in the population history, in the format:
    """
    recs = rec_str.strip().split('\n')

    return list(map(lambda x: list(map(int, x.split(' '))), recs))


def parse_mutants(mutant_data):
    """
    Returns each mutation event as a tuple, read form simuPOP output string
    """
    sci_notation_to_int = lambda x: int(float(x))
    for line in mutant_data.split('\n'):
        # a trailing \n will lead to an empty string
        if not line:
            continue
        (gen, loc, ploidy, a1, a2, ID) = line.split('\t')
        loc = sci_notation_to_int(loc)
        a1 = sci_notation_to_int(a1)
        a2 = sci_notation_to_int(a2)
        ID = sci_notation_to_int(ID)
        yield gen, loc, ploidy, a1, a2, ID


def store_mutants(mutant_data):
    """
    Returns simuPOP mutation output as a structured numpy array
    """
    dtypes = [('gen', int), ('loc', int), ('chrom', int), ('a1', int),
              ('a2', int), ('ID', int)]
    muts = list(parse_mutants(mutant_data))

    if len(muts) > 0:
        mut_array = np.core.records.fromrecords(muts, dtype=dtypes)
    else:
        mut_array = np.array([], dtype=dtypes)

    return mut_array


