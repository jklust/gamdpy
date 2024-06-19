""" Example of a binary LJ simulation using rumdpy.

NVT simulation of the Kob-Andersen mixture, starting from a FCC crystal using a temperature ramp.

"""
from __future__ import annotations
import rumdpy as rp
import numpy as np
import pandas as pd
import h5py

rho = 1.200
# Setup configuration: FCC crystal
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=rho, T=1.6)
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
steps_per_block = 1*1024  # ... each of this many steps
running_time = dt*num_blocks*steps_per_block
temperature = 0.800
filename = 'Data/KABLJ_Rho1.20_T0.800.h5'

print('High Temperature followed by cooling:')
Ttarget_function = rp.make_function_ramp(value0=2.000,       x0=running_time*(1/2), 
                                         value1=temperature, x1=running_time*(1/1))
integrator = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)
sim = rp.Simulation(configuration, pairpot, integrator, 
                    num_blocks=num_blocks, steps_per_block=steps_per_block, 
                    storage=filename) 
for block in sim.blocks():
    print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

print('Equilibration:')
integrator = rp.integrators.NVT(temperature, tau=0.2, dt=dt)
sim = rp.Simulation(configuration, pairpot, integrator, 
                    num_blocks=num_blocks, steps_per_block=steps_per_block, 
                    storage=filename)
for block in sim.blocks():
    print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

print('Production:')
for block in sim.blocks():
    print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())

columns = ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol']
with h5py.File(filename, "r") as f:
       data = np.array(rp.extract_scalars(f, columns, first_block=1))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U']))*dt*sim.steps_between_output # should be build in

mu = np.mean(df['U'])/configuration.N
vu = np.var(df['U'])/configuration.N
mw = np.mean(df['W'])/configuration.N
vw = np.var(df['W'])/configuration.N

print(f'Potential energy: {mu:.4f}, {vu:.4f}')
print(f'Virial            {mw:.4f}, {vw:.4f}')
if rho==1.200 and temperature==0.800:
     print('Testing complience with Rumd3')
     assert abs(mu + 6.346) < 0.01
     assert abs(mw - 5.534) < 0.01
     print('Passed')

rp.plot_scalars(df, configuration.N,  configuration.D, figsize=(10,8), block=True)

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 16 -o msd_KABLJ.pdf Data/KABLJ_Rho*T*.h5
