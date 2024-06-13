""" Example of computing the structure factor of a Lennard-Jones liquid.
    S(𝐪) = 1/N * |sum_{i=1}^{N} exp(-i𝐪•𝐫)|^2
    where 𝐪 is the wave vector, 𝐫 is the position of the particles, and N is the number of particles.
    The 𝐪 vectors are given by 𝐪 = 2π𝐧/L, where 𝐧 is a vector of integers and L is the box size.

    Below we compute for several 𝐧 vectors and plot the structure factor.

"""
import itertools
import time

import matplotlib.pyplot as plt

import rumdpy as rp
import numpy as np

np.random.seed(2024)

configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=3.0)

# Make ideal gas, or add small random displacements to the particles
dx = 'ideal gas'  # | 0.1
if dx == 'ideal gas':
    L = configuration.simbox.lengths
    configuration['r'] = np.random.rand(*configuration['r'].shape) * L
else:
    print(configuration['r'][0][0])
    configuration['r'] += dx * np.random.rand(*configuration['r'].shape)-dx/2
    print(configuration['r'][0][0])

# Get configuration data and set up q vectors
L = configuration.simbox.lengths
D = configuration.D
N = configuration.N
r = configuration['r']
# n_vectors: [[0, 0, 0], [1, 0, 0], [2, 0, 0], ..., [5, 5, 5]]
n_max = 12
n_vectors = np.array(list(itertools.product(range(n_max), repeat=D)), dtype=int)
# Remove the first vector [0, 0, 0]
n_vectors = n_vectors[1:]
q = 2 * np.pi * n_vectors / L
q_lengths = np.linalg.norm(q, axis=1)

# Compute the structure factor
r_dot_q = np.dot(r, q.T)
tic = time.perf_counter()
S_of_q = np.abs(np.sum(np.exp(1j * r_dot_q), axis=0)) ** 2 / N
toc = time.perf_counter()
print(f"Elapsed time: {toc - tic:.2f} s")

# Bin the structure factor to reduce noise
q_min_for_binning = 1.0
q_bin_width = 0.3
q_bins = np.arange(q_min_for_binning, q_lengths.max(), q_bin_width)
S_of_q_binned = np.zeros_like(q_bins)
for i, q_bin in enumerate(q_bins):
    mask = (q_lengths >= q_bin) & (q_lengths < q_bin + q_bin_width)
    S_of_q_binned[i] = np.mean(S_of_q[mask])

# Add un-binned and binned structure factors to the plot
S_of_q_unbinned = S_of_q[q_lengths <= q_min_for_binning]
S = np.append(S_of_q_unbinned, S_of_q_binned)
q = np.append(q_lengths[q_lengths <= q_min_for_binning], q_bins)

plt.figure()
plt.plot(q, S, 'o')
plt.xlabel('q')
plt.ylabel('S(q)')
plt.ylim(0, 3)
plt.show()

