import msprime
import simuPOP as sim
import numpy as np
import sys, os
import argparse


def generate_source_pops(args):
    ## Initialize admixed and source populations with two chromosomes per ind
    pop0_size = int(2 * args.Na * args.admixed_prop)
    pop1_size = 2 * args.Na - pop0_size

    population_configurations = [
            msprime.PopulationConfiguration(
                    sample_size=pop0_size,
                    initial_size=pop0_size,
                    growth_rate=0),
            msprime.PopulationConfiguration(
                    sample_size=pop1_size,
                    initial_size=pop1_size,
                    growth_rate=0)]

    ## Specify admixture event
    demographic_events = [
            msprime.MassMigration(
                    time=args.t_div,
                    source=0,
                    destination=1,
                    proportion=1.)]
            
    ## Coalescent simulation
    ts = msprime.simulate(
            population_configurations=population_configurations,
            demographic_events=demographic_events,
            recombination_rate=args.rho,
            length=args.length,
            mutation_rate=args.mu,
            Ne=args.Na)

    return ts


def msprime_hap_to_simuPOP(TreeSequence):
    """
    Takes msprime haplotypes and returns them in a format readable by simuPOP
    """
    haplotypes = TreeSequence.haplotypes()
    simuPOP_haps = [list(map(int, list(str(x)))) for x in haplotypes]

    return simuPOP_haps


def msprime_positions(TreeSequence):
    """ Returns position of mutations in TreeSequence """
    ##TODO Could be done in the haplotype loop above +t3
    return [site.position for site in TreeSequence.sites()]


def wf_init(haplotypes, positions):
    """
    Initializes a simuPOP population using the provided haplotypes
    """
    pop = sim.Population(size=[len(haplotypes)/2])
    pop.addChrom(positions, chromName='0')
    
    ## Set genotypes for each individual separately
    ##TODO Probably a more efficient way of setting genotypes +t2
    for ind, gen in zip(pop.individuals(), haplotypes):
        ind.setGenotype(gen)

    return pop


def main(args):
    ## Generate tree sequence
    ts = generate_source_pops(args)

    ## Parse haplotypes
    haplotypes = msprime_hap_to_simuPOP(ts)
    positions = msprime_positions(ts)

    ## Initialize simuPOP population
    pop = wf_init(haplotypes, positions)

    ## Evolve forward in time
    pop.evolve(gen=args.t_admix)

    ## Output genotypes
    genotypes = [ind.genotype() for ind in pop.individuals()]
    print(genotypes)


if __name__ == "__main__":
    args = argparse.Namespace(
            Na=10,
            t_admix=10,
            t_div=1000,
            admixed_prop=0.5,
            rho=1e-8,
            mu=1e-8,
            length=1e6
            )

    main(args)
