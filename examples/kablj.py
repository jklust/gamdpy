""" Example of a binary LJ simulation using rumdpy.

NVT simulation of the Kob-Andersen mixture, starting from a FCC crystal using a temperature ramp.

"""
from __future__ import annotations
import rumdpy as rp
import numpy as np
import pandas as pd
import os.path
import h5py
import matplotlib.pyplot as plt

rho = 1.200
# Setup configuration: FCC crystal
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=rho)
configuration.randomize_velocities(T=1.6)
configuration.ptype[::5] = 1     # Every fifth particle set to type 1 (4:1 mixture)
#configuration['r'][27,2] += 0.01 # Pertube z-coordinate of particle 27

# Setup pair potential: Binary Kob-Andersen LJ mixture.
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
dt = 0.004 # timestep 
num_blocks = 64              # Do simulation in this many 'blocks'
steps_per_block = 4*1024  # ... each of this many steps
running_time = dt*num_blocks*steps_per_block
temperature = 0.800
filename = 'Data/KABLJ_Rho1.20_T0.800.h5'

print('High Temperature followed by cooling and equilibration:')
Ttarget_function = rp.make_function_ramp(value0=2.000,       x0=running_time*(1/8), 
                                         value1=temperature, x1=running_time*(1/4))
integrator = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)
sim = rp.Simulation(configuration, pairpot, integrator,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename) 
for block in sim.timeblocks():
    print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

print('Production:')
integrator = rp.integrators.NVT(temperature, tau=0.2, dt=dt)
sim = rp.Simulation(configuration, pairpot, integrator,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename)
for block in sim.timeblocks():
    print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

columns = ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol']
with h5py.File(filename, "r") as f:
       data = np.array(rp.extract_scalars(f, columns, first_block=1))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U']))*dt*sim.steps_between_output # should be build in

mu = np.mean(df['U'])/configuration.N
mw = np.mean(df['W'])/configuration.N
cvex = np.var(df['U'])/temperature**2/configuration.N

print('rumdpy:')
print(f'Potential energy:     {mu:.4f}')
print(f'Excess heat capacity: {cvex:.3f}')
print(f'Virial                {mw:.4f}')

if rho==1.200 and temperature==0.800:
       mu3 = -6.346
       mw3 =  5.534
       cvex3 = 0.0001089086505*10000/0.8**2
       print('\nRumd3:')
       print(f'Potential energy:     {mu3:.4f}')
       print(f'Excess heat capacity: {cvex3:.3f}')
       print(f'Virial                {mw3:.4f}')

rp.plot_scalars(df, configuration.N,  configuration.D, figsize=(10,8), block=False)

with h5py.File(filename, "r") as f:
       dyn = rp.tools.calc_dynamics(f, first_block=0, qvalues=[7.5, 5.5])

fig, axs = plt.subplots(1, 1, figsize=(6,4))
axs.loglog(dyn['times'], dyn['msd'], '.-', label=['A (rumdpy)', 'B (rumdpy)'])
axs.set_xlabel('Time')
axs.set_ylabel('MSD')

rumd3filename = f'Data/KABLJ_msd_R{rho:.3f}_T{temperature:.3f}_rumd3.dat'
if os.path.isfile(rumd3filename):
     msd3 = np.loadtxt(rumd3filename)
     axs.loglog(msd3[:21,0], msd3[:21,1:], '--', label=['A (rumd3)', 'B (rumd3)'])

axs.legend()
plt.show(block=True)

if rho==1.200 and temperature==0.800:
     print('Testing complience with Rumd3:')
     assert abs(mu - mu3) < 0.01
     assert abs(cvex - cvex3) < 0.1
     assert abs(mw - mw3) < 0.03
     print('Passed')
