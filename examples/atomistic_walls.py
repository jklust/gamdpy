""" Example of a nanoslit pore simulation using tethered LJ particles 

    The particles are tethered with a Hooke spring.   

    Wall density is set to 1.0 and fluid density to 0.8 - this is achieved by inclusion of a dummy particle
    Currently the entire system is thermostated. 

    Initial and final configurations are saved in xyz format for easy inspection in vmd
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
nxUnits, nyUnits, nzUnits, rhoWall, rhoFluid = 6, 6, 10, 1.0, 0.8;
configuration = rp.make_configuration_fcc(nxUnits, nyUnits, nzUnits, rhoWall)

# Fluid particles have type '0', wall particles '1', dummy particles '2'
nwall, npart = 0, configuration.N
hlz = 0.5*configuration.simbox.lengths[2]
for n in range(npart):
    if configuration['r'][n][2] + hlz < 5.0:
        configuration.ptype[n] = 1
        nwall = nwall + 1

nfluid = np.sum( configuration.ptype==0 )
nfluidWanted = int(nfluid*rhoFluid/rhoWall)

while nfluid > nfluidWanted:
    idx = random.randint(0,npart-1)
    if configuration.ptype[idx] == 0:
        configuration.ptype[idx] = 2
        nfluid = nfluid - 1

savexyz(configuration, "initial.xyz")

# Tether specifications. 
# Tether-parameters: [x0, y0, z0, kspring] ; Index array: [row index in param, particle/atom index]  
springconstant = 1000.0
indices_array = []
tether_parameters = []
counter = 0
for n in range(npart):
    if configuration.ptype[n]==1:
        indices_array.append([counter, n])
        counter = counter + 1
        pos =  configuration['r'][n]
        tether_parameters.append( [pos[0], pos[1], pos[2], springconstant] )

tether = rp.Tether(tether_parameters, indices_array, verbose=False)

# Set the pair interactions
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]] 
eps = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
cut = np.array(sig)*2.5 
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Temperature
configuration.randomize_velocities(T=0.7)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)

# Some compute plan settings for old cards 
compute_plan = rp.get_default_compute_plan(configuration)
compute_plan['gridsync']=False
compute_plan['tp']=2 

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, [pairpot, tether], integrator,
                    num_timeblocks=12, steps_per_timeblock=6*1024,
                    storage='LJ_T0.70.h5', compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))

print(sim.summary())

savexyz(configuration, "final.xyz")
