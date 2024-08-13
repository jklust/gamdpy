""" Simulate a system of asymmetric dumbbells (ASD) with a Lennard-Jones potential and a harmonic bond potential. """

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

import rumdpy as rp

# Generate configuration with a FCC lattice
# rho = 0.932 # Molecular density
rho = 1.863  # Atomic density

configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[6, 6, 6], rho=rho)
configuration['m'] = 1.0
B_particles = range(1, configuration.N, 2)
configuration.ptype[B_particles] = 1  # Setting particle type of B particles
configuration['m'][B_particles] = 0.195  # Setting masses of B particles
configuration.randomize_velocities(T=1.44)

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
steps_per_block = 1024  # ... each of this many steps (increase for better statistics)
running_time = dt * num_blocks * steps_per_block
temperature = 0.465
Ttarget_function = rp.make_function_ramp(value0=10.000, x0=running_time * (1 / 8),
                                         value1=temperature, x1=running_time * (1 / 4))
integrator0 = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

compute_plan = rp.get_default_compute_plan(configuration)
print(compute_plan)
#compute_plan['tp'] = 6

sim = rp.Simulation(configuration, [pair_pot, bonds], integrator0,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    steps_between_momentum_reset=100,
                    compute_plan=compute_plan, storage='memory')

print('High Temperature followed by cooling and equilibration:')
for block in sim.timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())
sim.output.close()

runtime_action = 128
#runtime_action=1024*8 # to see effect of momentum resetting
#runtime_action=0 # Turn off momentum resetting (at own risk!)

integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)
sim = rp.Simulation(configuration, [pair_pot, bonds], integrator,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    steps_between_momentum_reset=runtime_action,
                    compute_plan=compute_plan, storage='memory')

# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = rp.CalculatorRadialDistribution(configuration, num_bins=1000)

print('Production:')
for block in sim.timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
    calc_rdf.update()
print(sim.summary())

# scalars
columns = ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol', 'Px', 'Py', 'Pz']
data = np.array(rp.extract_scalars(sim.output, columns, first_block=0))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U'])) * dt * sim.output_calculator.steps_between_output  # should be build in
if callable(temperature):
    df['Ttarget'] = numba.vectorize(temperature)(np.array(df['t']))
rp.plot_scalars(df, configuration.N, configuration.D, figsize=(10, 8), block=False)

dynamics = rp.tools.calc_dynamics(sim.output, 0, qvalues=(7.5, 7.5))
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
rdf['rdf'] = np.mean(rdf['rdf'], axis=0)
fig, axs = plt.subplots(1, 1, figsize=(8, 4))
axs.set_ylabel('RDF')
axs.set_xlabel('Distance')
axs.grid(linestyle='--', alpha=0.5)
axs.plot(rdf['distances'], rdf['rdf'], '-')
axs.set_xlim([0.5, 3.5])
plt.show(block=True)

