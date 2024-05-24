""" Example of performing several simulation in one go using rumdpy.

An isomorph is traced out using the gamma method. The script demomstrates
the possibility of keeping the output of the simulation in memory (storage='memory').
This is usefull when a lot of short simulations are performed.

To plot the results do: 
python plot_isomorph_dynamics.pdf
python plot_isomorph_rdf.pdf

For a simpler script performing multiple simulations, see isochore.py

"""

import rumdpy as rp
import numpy as np
import pickle

# Setup pair potential.
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)


T = 0.8

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.84, T=2*T)
    
# Setup integrators
integrator1 = rp.integrators.NVT(temperature=T, tau=0.2, dt=0.0025)
integrator2 = rp.integrators.NVT(temperature=T, tau=0.2, dt=0.0025)

# Setup Simulations
sim1 = rp.Simulation(configuration, pairpot, integrator1, 
                    num_blocks=4, steps_per_block=512,
                    scalar_output=1, 
                    storage='memory') 

sim2 = rp.Simulation(configuration, pairpot, integrator2, 
                    num_blocks=4, steps_per_block=512,
                    scalar_output=1,
                    storage='memory') 
    

print(configuration['r'][1])
print('Integrator1, Equilibration:', end='\t')
for block in sim1.blocks():
    pass
print(sim1.status(per_particle=True))
U1, K1 = rp.extract_scalars(sim1.output, ['U', 'K'], first_block=0)
E1 = U1 + K1

print('Integrator1, Production:', end='\t')
for block in sim1.blocks():
    pass
print(sim1.status(per_particle=True))
U2, K2 = rp.extract_scalars(sim1.output, ['U', 'K'], first_block=0)
E2 = U2 + K2 

print('Integrator2, Production:', end='\t')
for block in sim2.blocks():
    pass
print(sim2.status(per_particle=True))

U3, K3 = rp.extract_scalars(sim2.output, ['U', 'K'], first_block=0)
E3 = U3 + K3 

import matplotlib.pyplot as plt

plt.plot(U1, '.-', label='Integrator1, Equilibration')
plt.plot(U2, '.-', label='Integrator1, Production')
plt.plot(U3, '.-', label='Integrator2, Production')
plt.legend()
plt.show()
