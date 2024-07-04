""" Example using tethered LJ particles 

    Particles with label 1 and 2 are tethered with different Hooke springs.  
    All particles are integrated forward in time with the NVT integrator.
    Density of fluid/free particles and tethered particles the same.
    Initial and final configurations are saved in xyz format for easy inspection in vmd (guess a flag...)
"""

import numpy as np
import random 
import rumdpy as rp

def savexyz(configuration, filestr):
    
    configuration.get_vector('r')
    npart = configuration.N

    fp =  open(filestr,"w")
    fp.write("%d\n" % (npart))
    fp.write("Test for tethered particles\n")
    for n in range(npart):
        fp.write("%d %f %f %f\n" % 
                 (configuration.ptype[n], configuration['r'][n][0], configuration['r'][n][1], configuration['r'][n][2])) 
    fp.close()


# Setup a default fcc configuration
nxUnits, nyUnits, nzUnits, rho = 6, 6, 10, 1.0
configuration = rp.make_configuration_fcc(nxUnits, nyUnits, nzUnits, rho)

# Fluid/free particles have type '0', tethered particles '1' and '2'
for n in range(configuration.N):
    if  configuration['r'][n][2] > -3 and configuration['r'][n][2] < -1:
        configuration.ptype[n] = 1
    elif configuration['r'][n][0] > -1 and configuration['r'][n][0] < 1:
        configuration.ptype[n] = 2

savexyz(configuration, "initial.xyz")

# Tether specifications. 
tether = rp.Tether([1, 2], [300, 500], configuration, verbose=True)

# Set the pair interactions
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] 
eps = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
cut = np.array(sig)*2.5 
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Temperature
configuration.randomize_velocities(T=2.0)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=2.0, tau=0.2, dt=0.005)

# Some compute plan settings for old cards 
compute_plan = rp.get_default_compute_plan(configuration)
compute_plan['gridsync']=False
compute_plan['tp']=2 

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, [pairpot, tether], integrator,
                    num_timeblocks=100, steps_per_timeblock=128,
                    steps_between_momentum_reset=0, storage='LJ_T0.70.h5', compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))

print(sim.summary())

savexyz(configuration, "final.xyz")
