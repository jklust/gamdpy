import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py
import os

filename = 'ZJW-2004-parameters-20250821.txt'

with open(filename, 'r') as f:
    header_line = f.readline().strip()

column_names = header_line.lstrip('#').split()

df = pd.read_csv(filename, sep='\s+', skiprows=1, names=column_names)
params = df['Cu'].to_list()
cut = 6.0
params.append(cut)
params_array = np.array(params)

rho = 0.085 # number density in inverse cubic Angstrom
TK = 2500 # Temperature in Kelvin
# convert temp to eV/kB units

Aa = 1.e-10 # Angstrom in m
Na = 6.02214076e23
gPerMole = 1.e-3/Na
Da = 1.66053906892e-27
eV = 1.602176634e-19
kB = 1.380649e-23
T = TK * kB/eV
print("Temperature in eV", T)

print(params_array)

conf = gp.Configuration(D=3)
conf.make_lattice(gp.unit_cells.FCC, cells=[8,8,8], rho=rho)
conf['m'] = 63.546
conf.randomize_velocities(temperature=T)


eam_pot = gp.EAM_ZJW_2004([params_array], max_num_nbs=1000)

integrator = gp.NVT(temperature=T, tau=5.0, dt=0.1)

runtime_actions = [gp.TrajectorySaver(),
                   gp.MomentumReset(100),
                   gp.ScalarSaver(16),
                   gp.RestartSaver()]

sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=15, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage='test_eam.h5')




for block in sim.run_timeblocks():
    print(f"{sim.status(per_particle=True)}")


with h5py.File(f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5", "w") as f:
    conf.save(f, group_name='configuration', mode='w')
    
print(sim.summary())


dump_filename = f'Cu-liquid-rho{rho:.4f}-T{T:.4f}.lammps'
if os.path.exists(dump_filename):
    os.remove(dump_filename)


# when writing to LAMMPS dump file want to:
# (1) increment the velocities by a half time step
# (2) convert to LAMMPS "metal" units where the time unit is picosecond
time_unit_in_picoseconds = Aa * math.sqrt(Da/eV)*1e12
print('time unit', time_unit_in_picoseconds)
unit_ratios = {'positions-box' : 1., 'velocities': 1./time_unit_in_picoseconds, 'forces':1.}

dt = integrator.get_params(sim.configuration, sim.interactions_params)[0]
#dt = 0.01 # for testing, to be consistent with the LAMMPS script
print('Increment velocities by a half time-step for saving to LAMMPS dump file')
print(f'dt={dt}')
sim.configuration['v'] += (sim.configuration['f']/sim.configuration['m'][:, np.newaxis]) * (dt/2.)

lmp_dump = gp.configuration_to_lammps(sim.configuration, timestep=0, unit_rescale=unit_ratios)
print(lmp_dump, file=open(dump_filename, 'a'))

