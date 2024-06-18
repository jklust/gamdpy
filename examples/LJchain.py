import numpy as np
import rumdpy as rp
import numba
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt

# Generate configuration with a FCC lattice
rho = 0.8442
configuration = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=rho,  T=1.44)  

# Make bonds
bond_potential = rp.harmonic_bond_function
potential_params_list = [[1.12, 3000.], [1.00, 3000.], [1.12, 3000.]]
fourth = np.arange(0,configuration.N,4)
bond_particles_list = [np.array((fourth, fourth+1)).T, np.array((fourth+1, fourth+2)).T, np.array((fourth+2, fourth+3)).T] 
bonds = rp.Bonds(bond_potential, potential_params_list, bond_particles_list)

# Make pair potential
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Make integrator
dt = 0.002 # timestep 
num_blocks = 128               # Do simulation in this many 'blocks'
steps_per_block = 1024*4      # ... each of this many steps
running_time = dt*num_blocks*steps_per_block
temperature = 2.5

integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt) 

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
compute_plan = rp.get_default_compute_plan(configuration)
print(compute_plan)
compute_plan['tp'] = 6

sim = rp.Simulation(configuration, [pairpot, bonds], integrator,
#sim = rp.Simulation(configuration, [pairpot, ], integrator,
                    num_blocks=num_blocks, steps_per_block=steps_per_block,
                    compute_plan=compute_plan, storage='memory')

# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = rp.CalculatorRadialDistribution(configuration, num_bins=1000)

print('Equilibration:')
for block in sim.blocks():
    print('.', end='', flush=True)
print()
print(sim.summary())

print('Production:')
for block in sim.blocks():
    if block%10==0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
    calc_rdf.update()
print(sim.summary())

# scalars
columns = ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol']
data = np.array(rp.extract_scalars(sim.output, columns, first_block=0))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U']))*dt*sim.steps_between_output # should be build in
if callable(temperature):
    df['Ttarget'] = numba.vectorize(temperature)(np.array(df['t']))
rp.plot_scalars(df, configuration.N,  configuration.D, figsize=(10,8), block=False)

dynamics = rp.tools.calc_dynamics(sim.output, 0, qvalues=7.5*rho**(1/3))
fig, axs = plt.subplots(3, 1, figsize=(8,9), sharex=True)
fig.subplots_adjust(hspace=0.00) # Remove vertical space between axes
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

rdf = calc_rdf.read()
rdf['rdf'] = np.mean(rdf['rdf'], axis=0)
fig, axs = plt.subplots(1, 1, figsize=(8,4))
axs.set_ylabel('RDF')
axs.set_xlabel('Distance')
axs.grid(linestyle='--', alpha=0.5)
axs.plot(rdf['distances'], rdf['rdf'], '-')
axs.set_xlim([0.5, 3.5])
plt.show(block=True)


