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


def compute_structure_factor(conf, verbose=False):
    # Get configuration data and set up q vectors
    L = conf.simbox.lengths
    D = conf.D
    N = conf.N
    r = conf['r']
    # n_vectors: [[0, 0, 0], [1, 0, 0], [2, 0, 0], ..., [18, 18, 18]]
    n_max = 24
    n_vectors = np.array(list(itertools.product(range(n_max), repeat=D)), dtype=int)
    # Remove the first vector [0, 0, 0]
    n_vectors = n_vectors[1:]
    # Remove n_vectors where the length is greater than n_max
    n_vectors = n_vectors[np.linalg.norm(n_vectors, axis=1) < n_max]
    q_vectors = 2 * np.pi * n_vectors / L
    q_len = np.linalg.norm(q_vectors, axis=1)

    # Compute the structure factor
    r_dot_q = np.dot(r, q_vectors.T)
    tic = time.perf_counter()
    structure_factor = np.abs(np.sum(np.exp(1j * r_dot_q), axis=0)) ** 2 / N
    toc = time.perf_counter()
    wall_clock_time = toc - tic
    if verbose:
        print(f"Wall-clock time to compute S(q): {wall_clock_time:.4f} s")

    return q_len, structure_factor, wall_clock_time


# Bin the structure factor to reduce noise
def binning(structure_factor, q_lengths):
    q_min_for_binning: float = 1.0
    q_bin_width: float = 0.2
    q_bins = np.arange(q_min_for_binning, q_lengths.max(), q_bin_width)
    S_of_q_binned = np.zeros_like(q_bins)
    q_binned = np.zeros_like(q_bins)
    for i, q_bin in enumerate(q_bins):
        mask = (q_lengths >= q_bin) & (q_lengths < q_bin + q_bin_width)
        S_of_q_binned[i] = np.mean(structure_factor[mask])
        q_binned[i] = np.mean(q_lengths[mask])

    # Add un-binned and binned structure factors to the plot
    S_of_q_unbinned = structure_factor[q_lengths <= q_min_for_binning]
    S = np.append(S_of_q_unbinned, S_of_q_binned)
    q = np.append(q_lengths[q_lengths <= q_min_for_binning], q_binned)
    return q, S

# Setup simulation
temperature = 2.0
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=temperature*2)
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_potential = rp.PairPotential2(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)
sim = rp.Simulation(configuration, pair_potential, integrator,
                    steps_per_block=512, num_blocks=128, storage='memory')

print("Equilibration run")
sim.run()

print("Production run")
structure_factor_list = []
wall_clock_times = []
for block in sim.blocks():
    print(sim.status(per_particle=True))
    q_lengths, S_of_q, wall_clock_time = compute_structure_factor(sim.configuration, verbose=True)
    q, S = binning(S_of_q, q_lengths)
    structure_factor_list.append(S)
    wall_clock_times.append(wall_clock_time)

# Average the structure factor
S = np.mean(structure_factor_list, axis=0)

plt.figure()
plt.plot(q, S, 'o')
plt.yscale('log')
plt.xlabel(r'$|q|$')
plt.ylabel('$S(q)$')
plt.ylim(1e-2, 3)
plt.xlim(0, 14)
plt.show()
