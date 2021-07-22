import numpy as np
from numpy.core.numeric import full
import nn_utility_functions as nfunc
import matplotlib.pyplot as plt

path = '/home/ngc/Documents/GitHub/neural_nets/eschbach_2020/data'

names, avg_connectivity = nfunc.filtered_average_connectivity(path, 3, 9)

all_names, full_connectivity = nfunc.both_hemispheres_connectivity(
	path,
	cutoff = 3
	)

plt.figure('avg')
plt.imshow(avg_connectivity[80:120, :], cmap='gray')
plt.figure('full')
plt.imshow(full_connectivity[80:120, :], cmap='gray')
plt.show()

lines = plt.figure('lines')
plt.plot(avg_connectivity[80, :], label='avg')
plt.plot(full_connectivity[80], label='full')
plt.legend()
plt.show()

nfunc.sparsity_index(avg_connectivity)
nfunc.sparsity_index(full_connectivity)