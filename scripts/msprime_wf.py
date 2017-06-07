import msprime
import simuPOP as sim
from collections import defaultdict
import numpy as np
import sys, os
import attr
import argparse


def generate_source_pops():
    args = argparse.Namespace(
            Na=100,
            t_admix=10,
            t_div=1000,
            admixed_prop=0.5,
            rho=1e-8,
            mu=1e-8,
            length=1e7
            )

    ## Initialize admixed and source populations
    pop0_size = int(args.Na * args.admixed_prop)
    pop1_size = args.Na - pop0_size

    population_configurations = [
            msprime.PopulationConfiguration(
                    sample_size=pop0_size,
                    initial_size=pop0_size,
                    growth_rate=0),
            msprime.PopulationConfiguration(
                    sample_size=pop1_size,
                    initial_size=pop1_size,
                    growth_rate=0)
            ]

    ## Specify admixture event
    demographic_events = [
            msprime.MassMigration(time=args.t_div, source=0, destination=1,
                                    proportion=1.),
            ]
            
    ## Coalescent simulation
    ts = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            recombination_rate=args.rho, length=args.length,
                            mutation_rate=args.mu, Ne=args.Na)

    return ts


def msprime_hap_to_simuPOP(TreeSequence, sort=None):
    """
    Takes msprime haplotypes and returns them in a format readable by simuPOP
    """
    haplotypes = TreeSequence.haplotypes()
    G = [list(map(int, list(str(x)))) for x in haplotypes]

    return G


def wf_init(haplotypes):
    """
    Initializes a simuPOP population using the provided haplotypes
    """
    n_loci = len(haplotypes[0])
    pop = sim.Population(size=[len(haplotypes)/2], loci=[n_loci])
    
    ## Set genotypes for each individual separately
    for ind, gen in zip(pop.individuals(), haplotypes):
        ind.setGenotype(gen)

    return pop

