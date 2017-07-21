# msprime-abc
Demographic inference and more, using msprime and simuPOP

## wf\_tree.py
### Known Issues:
* ~~No mutations~~ - Simple dialleleic mutations using parameter *mu* stored in *self.muts*
* Fixed population size per generation
* Genotypes saved for all individuals in the forward simulations, rather than coalescent nodes only
* Generates intermediate text file containing genotype data before writing as hdf5 file

### Notes:
* *L*, *rho*, and *mu* are in units of base pairs
* Infinite-sites mutations with recombination doesn't seem simple to implement in simuPOP. Could specify loci under selection in forward simulations then throw down neutral mutations on trees afterwards to get full genotypes
* Disk usage estimate: 1000 inds, 1000 loci, 100 gens writes genotypes in plain text file ~500MB along with smaller hdf5 file

## hybrid\_sim.py
### Usage:
* Import into your script and provide ```hybrid_sim()``` with an arbitrary tree sequence to evolve forwards in time
* Function provided for extracting genotypes from simuPOP population - homologous chromosomes are concatenated
* Running the script in an IPython session will evolve an example tree sequence
* Initial demes are preserved, and arbitrary migration matrices can be specified

### Notes and current limitations:
* Fixed population size per generation for forward simulations
* Single chromosome with fixed number of loci
