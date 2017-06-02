import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys, os
import scipy.stats as stats
import scipy.spatial
import attr

import ancestry_blocks

##-------------------------------------------------------------------------
## Distance functions
##-------------------------------------------------------------------------

def bhattacharyya_coeff(mu0, sig0, mu1, sig1):
    """
    A measure of distance between two normal distributions.
    """
    smallnum = 1e-10
    sig0 = max(sig0, smallnum)
    sig1 = max(sig1, smallnum)

    term1 = 0.25 * np.log(0.25 * (sig0 / sig1 + sig1 / sig0 + 2))
    term2 = 0.25 * ((mu1 - mu0) ** 2 / (sig0 + sig1))

    return term1 + term2


def mean_var_dist(mu0, sig0, mu1, sig1):
    return np.max([(mu1 - mu0) ** 2, (sig1 - sig0) ** 2])


def mean_dist(mu0, mu1):
    return (mu1 - mu0) ** 2


def hist_distance(hist1, hist2):
    """
    Computes the distance between two histograms
    """
    ## Make sure the histograms have the same bin width, so we can match
    ## their supports.
    assert hist1[1][1] - hist1[1][0] == hist2[1][1] - hist2[1][0]

    hists = [list(hist1[0]), list(hist2[0])]
    l1 = len(hist1[0])
    l2 = len(hist2[0])

    if l1 != l2:
        order = np.argsort([l1, l2])
        hists[order[0]].extend([0 for i in range(np.abs(l2-l1))])

    # return scipy.stats.entropy(*hists)
    return scipy.spatial.distance.chebyshev(*hists)


def smooth_param_hist(params):
    """
    Generate smooth parameter likelihoods to facilitate resampling.
    """
    kde = stats.gaussian_kde(np.asarray(params).T)

    return kde


##-------------------------------------------------------------------------
## ABC class
##-------------------------------------------------------------------------
@attr.s
class ABC:
    chromlength = attr.ib()
    rho = attr.ib()
    Na = attr.ib()
    mingens = attr.ib()
    maxgens = attr.ib()
    ancestries = attr.ib()
    nbins = attr.ib(default=20)


    def __attrs_post_init__(self):
        self.colour_dict = self.make_colour_dict(self.ancestries)


    def set_obs_data(self, obs_data):
        self.obs_data = obs_data


    def make_colour_dict(self, ancestries):
        colour_list = plt.cm.Set1(np.linspace(0, 1, 12))

        return dict(zip(self.ancestries, colour_list))

    def theta_flat_prior(self):
        """
        Generate theta from a flat prior. For an autosome, format is:
        (admixture time, mig1prop)
        """
        t_admix = np.random.randint(self.mingens,self.maxgens)
        p1prop = np.random.random()
        theta = (t_admix, p1prop)

        return theta


    def theta_resample(self, kde, size=1, random=0.1):
        """
        Resamples theta according to which parameter values produced data
        within epsilon of the observation. We round ngens to the nearest
        integer.
        """
        resample = kde.resample(size).reshape((2))

        ## Keep proportions within bounds
        resample[0] = np.round(resample[0])
        resample[1] = np.max([0, resample[1]])
        resample[1] = np.min([1, resample[1]])
        # y = map(lambda x: max(0, x), resample[1])
        # y = map(lambda x: min(1, x), resample[1])
        ##@@ Only works for two parameters
        # resample_round = np.append(map(np.round, resample[0]), y)

        if random is not None:
            if np.random.random() < random:
                return self.theta_flat_pop()

        return resample


    def generate_data(self, theta):
        """
        Simulates population with given parameters and extracts ancestry
        tracts and proportions
        """
        t_admix, admixed_prop = theta

        ## Simulation args to pass
        args = argparse.Namespace(Na=self.Na,
                                t_admix=t_admix,
                                admixed_prop=admixed_prop,
                                length=self.chromlength,
                                rho=self.rho,
                                nbins=self.nbins)

        tracts, ancestry_props = ancestry_blocks.main(args)

        return tracts, ancestry_props


    def pop_abc(self, min_samples, epsilon):
        good_params = []
        num_samples = 0
        dists = []

        while num_samples < min_samples:
            theta = self.theta_flat_prior()
            sim_data = self.generate_data(theta)

            a_dist, h_dist = self.pop_data_dist(sim_data, self.obs_data)
            dists.append([a_dist, h_dist])

            if a_dist < epsilon[0] and h_dist < epsilon[1]:
                print("Accepted with distances", a_dist, h_dist)
                good_params.append(theta)
                num_samples += 1

        return np.asarray(good_params), dists


    def pop_data_dist(self, data0, data1):
        """
        Calculates the distance between two populations, described by their tract
        length distribution and ancestry proportion distribution.
        """
        tracts0, ancestry_props0 = data0
        tracts1, ancestry_props1 = data1

        h_dist = 0
        a_dist = 0
        for ancestry in tracts1.keys():
            ## Measure distance between tract-length histograms
            hist0 = tracts0[ancestry]
            hist1 = tracts1[ancestry]
            h_dist += hist_distance(hist0, hist1)

            ## Measure distance between ancestry proportions
            prop0 = ancestry_props0[ancestry]
            prop1 = ancestry_props1[ancestry]
            a_dist += mean_dist(prop0, prop1)

        # print("Distance components:", a_dist, h_dist)

        return a_dist, h_dist


