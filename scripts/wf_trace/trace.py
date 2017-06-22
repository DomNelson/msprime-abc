import numpy as np
import attr


def zero_index(lineage):
    """
    Convert 1-indexed signed lineage IDs into signed 0-indexed parent IDs,
    and assign unique IDs for each generation
    """
    ## Shift abs(ID) to abs(ID)-1
    lineage = lineage - np.sign(lineage)

    return lineage.astype(int)


def generation_index(lineage):
    """
    Return array where values denote the index in the generation prior that
    the individual inherited the allele from
    """
    ##TODO Currently depends on implementation detail of replicates +t1
    n_gens, n_inds, n_loci = lineage.shape
    x = np.ones((n_inds, n_loci))
    shift_rows = np.stack([x * i * n_inds for i in range(n_gens)])

    lineage = lineage - np.sign(lineage) * shift_rows

    return lineage.astype(int)


def parent_idx(lineage):
    """
    Converts ind IDs in lineage into the indices of parents in the previous
    generation
    """
    lineage = zero_index(lineage)
    lineage = unsign_ID(lineage)
    lineage = generation_index(lineage)

    return lineage


def unsign_ID(lineages):
    """
    Takes signed maternal/paternal chromosomal IDs and converts to unsigned,
    by doubling number of IDs
    """
    ## Update indices to match, remembering that sign indicates maternal
    ## or paternal inheritance
    pos_vals = np.sign(np.abs(lineages) + lineages)
    lineages = np.abs(lineages) * 2 + pos_vals

    return lineages


def active_alleles(masked_alleles):
    """
    Returns alleles which can be traced to the sampled generation, namely
    those which have not been masked
    """
    return masked_alleles[~masked_alleles.mask].data


def find_coalescence(allele_vec, inheritance_vec):
    """
    Finds coalescent events in the current allele state
    """
    alleles_to_track = active_alleles(inheritance_vec)

    for allele in set(alleles_to_track):
        inherits_ix = np.where(new_allele_vec == allele)[0]

        if len(active_alleles(allele_vec[inherits_ix])) > 1:
            yield allele, inherits_ix


def step_lineage(allele_vec, inheritance_vec):
    """
    Returns a new allele vector updated to account for inheritances and
    coalescences
    """
    new_alleles = allele_vec[inheritance_vec]

    return new_alleles


@attr.s
class ForwardTrees(object):
    ID = attr.ib()
    lineage = attr.ib()
    genotype = attr.ib()
    proband_labels = attr.ib()


    def __attrs_post_init__(self):
        self.parent_indices = parent_idx(self.lineage)
        self.n_gens, self.n_chroms, self.n_loci = self.lineage.shape


    def allele_history(self, allele_num):
        """
        Returns the forward-time inheritance paths for each generation, for
        the given allele
        """
        return np.ma.masked_array(self.parent_indices[:, :, allele_num])


    def ancestor(self, allele_num, gen, ix):
        """
        Returns the ancestors carrying the allele at the given generation
        """
        return self.lineage[:, :, allele_num][gen][ix]


    def trace_lineage(self, allele_num):
        """
        Follows alleles backwards through a forward-time inheritance simulation
        """
        allele_history = self.allele_history(allele_num)

        ## Initialize labels for alleles
        alleles = np.ma.masked_array(self.proband_labels)
        alleles.mask = np.zeros(alleles.shape)

        for i, inheritance_vec in enumerate(allele_history[::-1]):
            ## Follow alleles through inheritance vector, masking alleles that
            ## have coalesced
            new_alleles = step_lineage(alleles, inheritance_vec)

            for allele, inherits_ix in find_coalescence(alleles, new_alleles):

                yield allele, active_alleles(alleles[inherits_ix]), i

                ## Update mask with all except first allele coalescing
                new_alleles.mask[inherits_ix[1:]] = 1

            alleles = new_alleles
