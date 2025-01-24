""" Simulate a system of asymmetric dumbbells (ASD) with a Lennard-Jones potential and a harmonic bond potential. """

import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

import rumdpy as rp


# Specify state point
rho = 1.863  # Atomic density = Molecular density * 2
temperature = 0.465
filename = f'Data/ASD_rho{rho:.3f}_T{temperature:.3f}.h5'

# Generate two-component configuration with a FCC lattice 
configuration = rp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Ptot':True, 'Vol':True})
configuration.make_lattice(rp.unit_cells.FCC, cells=[6, 6, 6], rho=rho)
configuration['m'] = 1.0
B_particles = range(1, configuration.N, 2)
configuration.ptype[B_particles] = 1  # Setting particle type of B particles
configuration['m'][B_particles] = 0.195  # Setting masses of B particles
configuration.randomize_velocities(temperature=1.44)

# Make bonds
bond_potential = rp.harmonic_bond_function
bond_params = [[0.584, 3000.], ]  # Parameters for bond type 0, 1, 2 etc (here only 0)
bond_indices = [[i, i + 1, 0] for i in range(0, configuration.N - 1, 2)]  # dumbells: i(even) and i+1 bonded with type 0
bonds = rp.Bonds(bond_potential, bond_params, bond_indices)

# Make pair potential
# pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.000, 0.894],
       [0.894, 0.788]]
eps = [[1.000, 0.342],
       [0.342, 0.117]]
cut = np.array(sig) * 2.5
exclusions = bonds.get_exclusions(configuration)
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
dt = 0.002  # timestep
num_blocks = 64  # Do simulation in this many 'blocks'
steps_per_block = 1*1024  # ... each of this many steps (increase for better statistics)
running_time = dt * num_blocks * steps_per_block

Ttarget_function = rp.make_function_ramp(value0=10.000, x0=running_time * (1 / 8),
                                         value1=temperature, x1=running_time * (1 / 4))
integrator0 = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

sim = rp.Simulation(configuration, [pair_pot, bonds], integrator0,
                    runtime_actions=[rp.MomentumReset(100)],
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage='memory')

print('High Temperature followed by cooling and equilibration:')
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(sim.status(per_particle=True))
print(sim.summary())

print('Production:')
integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

runtime_actions = [rp.MomentumReset(100), 
                   rp.ConfigurationSaver(), 
                   rp.ScalarSaver(32, {'Fsq':True, 'lapU':True, 'Ptot':True}), ]

sim = rp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions, 
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename)

for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

output = rp.tools.TrajectoryIO(filename).get_h5()

# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = rp.CalculatorRadialDistribution(configuration, bins=1000)
positions = output["block"][:, :, 0, :, ]
for i in range(positions.shape[0]):
    pos = positions[i, -1, :, :]
    configuration["r"] = pos
    configuration.copy_to_device()
    calc_rdf.update()

# scalars
columns = ['U', 'W', 'K', 'Fsq', 'lapU', 'Vol', 'Px', 'Py', 'Pz']
data = np.array(rp.extract_scalars(output, columns, first_block=0))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U'])) * dt * output.attrs["steps_between_output"]  # should be build in
if callable(temperature):
    df['Ttarget'] = numba.vectorize(temperature)(np.array(df['t']))
rp.plot_scalars(df, configuration.N, configuration.D, figsize=(10, 8), block=False)

dynamics = rp.tools.calc_dynamics(output, 0, qvalues=(7.5, 7.5))
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
axs[0].set_ylabel('MSD')
axs[1].set_ylabel('Non Gaussian parameter')
axs[2].set_ylabel('Intermediate scattering function')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

axs[0].loglog(dynamics['times'], dynamics['msd'], 'o--')
axs[1].semilogx(dynamics['times'], dynamics['alpha2'], 'o--')
axs[2].semilogx(dynamics['times'], dynamics['Fs'], 'o--')
plt.show(block=False)

fig, axs = plt.subplots(1, 1, figsize=(8, 4))
axs.set_ylabel('Center-of-mass velocity')
axs.set_xlabel('Time')
axs.grid(linestyle='--', alpha=0.5)
total_mass = np.sum(configuration['m'])
for label in ['Px', 'Py', 'Pz']:
    axs.plot(df['t'].values, df[label].values / total_mass, '-', label=label + '/M')
axs.legend()
plt.show(block=False)

rdf = calc_rdf.read()
total_rdf = np.mean(rdf['rdf'], axis=0)
fig = plt.figure(figsize=(8, 7))
gs = gridspec.GridSpec(3, 2, wspace=.5, hspace=1)
ax0 = fig.add_subplot(gs[0, :])
ax0.set_ylabel('RDF')
ax0.set_xlabel('Distance')
ax0.grid(linestyle='--', alpha=0.5)
ax0.plot(rdf['distances'], total_rdf, '-')
ax0.set_xlim([0.5, 3.5])

for i in range(2):
    for j in range(2):
        rdf_ij = np.mean(rdf['rdf_ptype'][:, i, j, :], axis=0)
        ax = fig.add_subplot(gs[i+1, j])
        ax.set_ylabel('RDF')
        ax.set_xlabel('Distance')
        ax.grid(linestyle='--', alpha=0.5)
        ax.plot(rdf['distances'], rdf_ij, '-')
        ax.set_xlim([0.5, 3.5])
        ax.set_title(f"$g(r)$ between particle {i} and {j}")
plt.show(block=True)

