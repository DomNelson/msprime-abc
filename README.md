# msprime-abc
Demographic inference and more, using msprime and simuPOP

## wf\_tree.py
### Known Issues:
* ~~No mutations~~ - Simple dialleleic mutations using parameter *mu* stored in *self.muts*
* Fixed population size per generation
* No method to alter default initial genotypes
* Limited to single randomly-mating population
* Genotypes saved for all individuals in the forward simulations, rather than coalescent nodes only
* Raw text file output by simuPOP not cleaned up after writing data in hdf5 format

### Notes:
* *L*, *rho*, and *mu* are in units of base pairs
* Infinite-sites mutations with recombination doesn't seem simple to implement in simuPOP. Could specify loci under selection in forward simulations then throw down neutral mutations on trees afterwards to get full genotypes
* Disk usage estimate: 1000 inds, 1000 loci, 100 gens writes genotypes in plain text file ~500MB along with smaller hdf5 file
