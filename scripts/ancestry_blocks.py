import msprime
import numpy as np
import sys, os
import math

admixed_sample_size = 30
admixed_pop_size = 30
t_div = 1000
t_admix = 500
Ne = 1000

## Set parameters of admixture event
admixed_prop = 0.3
A_size = admixed_sample_size * admixed_prop
B_size = admixed_sample_size * 1 - admixed_prop

## Initialize admixed and source populations
population_configurations = [
        msprime.PopulationConfiguration(sample_size=0,
                                        initial_size=Ne,
                                        growth_rate=0),
        msprime.PopulationConfiguration(sample_size=admixed_sample_size,
                                        growth_rate=0)]

## Specify admixture event
demographic_events = [
        msprime.MassMigration(time=t_admix, source=1, destination=0,
                                proportion=admixed_prop),
        msprime.MassMigration(time=t_div, source=0, destination=1,
                                proportion=1.)]
        
dp = msprime.DemographyDebugger(
    Ne=Ne,
    population_configurations=population_configurations,
    demographic_events=demographic_events)
dp.print_history()

## Coalescent simulation
ts = msprime.simulate(population_configurations=population_configurations,
                        demographic_events=demographic_events,
                        recombination_rate=1e-8, length=1e4, Ne=Ne)

## Collect nodes from the source populations
ancestor_nodes = set()
for i, node in enumerate(ts.nodes()):
    if t_div > node.time and t_admix < node.time:
        ancestor_nodes.add(i)

## Collect ancestry tracts from one of the source populations
tracts = [e for e in ts.edgesets() if e.parent in ancestor_nodes]
