import sys,os
import numpy as np
import ABC
import time
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow


ancestries = ['red', 'blue']
chromlength = 1e8
rho = 1e-8
theta_0 = (15, 0.2)
Na = 100

epsilon = (0.05, 100)
min_samples = 20
mingens = 10
maxgens = 30

## Initialize ABC object
A = ABC.ABC(
        chromlength=chromlength,
        rho=rho,
        Na=Na,
        mingens=mingens,
        maxgens=maxgens,
        ancestries=ancestries)

## Generate observed data by simulating from theta_0
time1 = time.time()
obs_data = A.generate_data(theta_0)
print("Simulated in", time.time() - time1, "second")
A.set_obs_data(obs_data)

## Run ABC, drawing theta from uniform distribution
params, dists = A.pop_abc(min_samples, epsilon)

print("Selected params:")
print(params)

kde = ABC.smooth_param_hist(params)

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

