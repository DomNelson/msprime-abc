import sys,os
sys.path.append(os.path.expanduser('../sims/'))
import markov_sims as sims
import numpy as np
import ABC
import time
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow


ancestries = ['red', 'blue']
chromlength = 30
rho = 1
theta_0 = (1000, 0.2)

epsilon = 10
min_samples = 100
mingens = 2
maxgens = 50
steps = 1

## Initialize ABC object
A = ABC.ABC(ancestries, chromlength, rho, mingens, maxgens)

## Generate observed data by simulating from theta_0
time1 = time.time()
obs_data = A.generate_data(theta_0)
print "Simulated in", time.time() - time1, "second"
sys.exit()
A.set_obs_data(obs_data)

## Run ABC once with flat prior
prior = A.autosome_theta_flat_prior
prior_kwargs = {}

selected = []

params, dists = A.autosome_abc_hist(min_samples, epsilon, prior, prior_kwargs)
selected.append(params)

print "Selected params:"
print selected[-1]

epsilon = np.percentile(dists, 20)
print "New epsilon:", epsilon

kde = A.smooth_param_hist(selected[-1])

# Plot parameter estimates
x_range = range(mingens, maxgens + 1)
y_range = np.arange(0, 1, 1. / 50)
x, y = np.meshgrid(x_range, y_range)
assert x.shape == y.shape

space = []
for xval in x_range:
    row = []
    for yval in y_range:
        row.extend(kde.pdf([xval, yval]))
    space.append(row)

space = np.asarray(space)

plt.imshow(space)
plotfile = os.path.expanduser('~/temp/kde_' + str(0) + '.png')
plt.savefig(os.path.expanduser(plotfile))
plt.clf()

# Run in adaptive mode, increasing selection of previously high-selected
# regions of theta distribution.
for i in range(1, steps):
    ## ABC with theta drawn from posterior distribution of last step
    prior = A.theta_resample
    prior_kwargs = {'kde':kde}

    params, dists = A.autosome_abc_hist(min_samples, epsilon,
                                                        prior, prior_kwargs)
    selected.append(params)
    ##@@ This should be a parameter for the ABC method
    epsilon = np.percentile(dists, 20)
    print "New epsilon:", epsilon

    print selected[-1]

    ## Calculate new posterior distribution
    kde = A.smooth_param_hist(selected[-1])

    # Plot parameter estimates
    x_range = range(mingens, maxgens + 1)
    y_range = np.arange(0, 1, 1. / 50)
    x, y = np.meshgrid(x_range, y_range)
    assert x.shape == y.shape

    space = []
    for xval in x_range:
        row = []
        for yval in y_range:
            row.extend(kde.pdf([xval, yval]))
        space.append(row)

    space = np.asarray(space)

    plt.imshow(space)
    plotfile = os.path.expanduser('~/temp/kde_' + str(i) + '.png')
    plt.savefig(os.path.expanduser(plotfile))
    plt.clf()

# print np.asarray(selected)

## Plot parameter estimates
# x_range = range(mingens, maxgens + 1)
# y_range = np.arange(0, 1, 1. / 50)
# x, y = np.meshgrid(x_range, y_range)
# assert x.shape == y.shape
#
# space = []
# for xval in x_range:
#     row = []
#     for yval in y_range:
#         row.extend(kde.pdf([xval, yval]))
#     space.append(row)
#
# space = np.asarray(space)
#
# plt.imshow(space)
# plotfile = os.path.expanduser('~/temp/kde_' + str(i) + '.png')
# plt.savefig(os.path.expanduser(plotfile)
# plt.clf()
