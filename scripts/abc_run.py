import sys,os
import numpy as np
import ABC
import time
import matplotlib.pyplot as plt
import argparse

def plot_params(args, params):
    """ Plot parameter estimates """
    ## Smooth probability distribution
    kde = ABC.smooth_param_hist(params)

    x_range = range(args.min_gens, args.max_gens + 1)
    y_range = np.arange(0, 1, 1. / 100)
    x, y = np.meshgrid(x_range, y_range)
    assert x.shape == y.shape

    space = []
    for xval in x_range:
        row = []
        for yval in y_range:
            row.extend(kde.pdf([xval, yval]))
        space.append(row)

    space = np.asarray(space)

    plt.imshow(space, origin='lower')
    plt.xlabel('Admixture proportion')
    plt.ylabel('Admixture time (generations)')
    plt.title('Estimated posterior distribution\nof demographic parameters')
    plt.colorbar()
    plotfile = os.path.expanduser('~/temp/kde_' + str(0) + '.png')
    plt.savefig(os.path.expanduser(plotfile))
    plt.clf()


def main(args):
    ## Initialize ABC object
    A = ABC.ABC(
            chromlength=args.length,
            rho=args.rho,
            Na=args.Na,
            mingens=args.min_gens,
            maxgens=args.max_gens,
            ancestries=args.ancestries)

    ## Generate observed data by simulating from theta_0
    time1 = time.time()
    obs_data = A.generate_data(args.obs_params)
    print("Simulated in", time.time() - time1, "second")
    A.set_obs_data(obs_data)

    ## Run ABC, drawing theta from uniform distribution
    params, dists = A.pop_abc(args.min_samples, args.epsilon)

    print("Selected params:")
    print(params)

    plot_params(args, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancestries", help="Ordered list of ancestries " +\
            "in format anc1,anc2,...,ancN", required=True)
    parser.add_argument("--length", help="Length in base pairs of observed" +\
            " data", type=int, required=True)
    parser.add_argument("--rho", help="Per-base pair recombination rate, " +\
            "default=1e-8", type=float, default=1e-8)
    parser.add_argument("--Na", help="Admixed population size, default=100",
            type=int, default=100)
    parser.add_argument("--epsilon", help="Maximum distance from observed " +\
            " data for a simulation to be accepted, in the format " +\
            "ancestry_dist,tractlen_dist", required=True)
    parser.add_argument("--min_samples", help="Minimum number of accepted " +\
            " simulations to perform, default=20", type=int, default=20)
    parser.add_argument("--min_gens", help="Minimum number of generations " +\
            "to simulate, default=0", type=int, default=0)
    parser.add_argument("--max_gens", help="Maximum number of generations " +\
            "to simulate, default=30", type=int, default=30)
    parser.add_argument("--obs_params", help="Parameters with which to " +\
            "simulate 'observed' data for testing, with format " +\
            "admixture_prop,admixture_time", required=True)

    args = parser.parse_args()

    ## Convert args to expected type
    args.ancestries = args.ancestries.split(',')
    args.epsilon = tuple(map(float, args.epsilon.split(',')))
    args.obs_params = tuple(map(float, args.obs_params.split(',')))

    main(args)
