""" Example of a nanoslit pore simulation using tethered LJ particles 

    The particles are tethered with a Hooke spring. The wall particles interact with a relaxation device.   
    All particles are integrated forward in time with the NVE integrator

    Wall density is set to 1.0 and fluid density to 0.8 - this is achieved by inclusion of a dummy particle 

    Initial and final configurations are saved in xyz format for easy inspection in vmd
"""

import numpy as np
import random 
import rumdpy as rp


# Setup a default fcc configuration
nxUnits, nyUnits, nzUnits, rhoWall, rhoFluid = 6, 6, 10, 1.2, 0.7;
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

rp.tools.save_configuration(configuration, "initial.xyz")

# Tether specifications. 
# Alternative instanciation tether=rp.Tether(<index array>, <tether params>, verbose=False)
# where  Index array: [row index in param, particle/atom index], tether parameters: [x0, y0, z0, kspring] 
tether = rp.Tether([1], [300.0], configuration)

# Temp relaxation for wall particles
# Altnative instanciation  relax = rp.Relaxtemp(<relax_parameters>, <indices_array>, verbose=False)
# where relax parameters: [Tdesired, tau (characteristic relax time 0<tau<<1)] index as tether
relax = rp.Relaxtemp([1],[0.01],[2.0], configuration)

# Set the pair interactions
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]] 
eps = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
cut = np.array(sig)*2.5 
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Temperature
configuration.randomize_velocities(T=2.0)

# Setup integrator: NVT
integrator = rp.integrators.NVE(dt=0.005)

# Some compute plan settings for old cards 
compute_plan = rp.get_default_compute_plan(configuration)
compute_plan['gridsync']=True
compute_plan['tp']=6 

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, [pairpot, tether, relax], integrator,
                    num_timeblocks=200, steps_per_timeblock=64,
                    steps_between_momentum_reset=0, storage='LJ_T0.70.h5', compute_plan=compute_plan)

prof = rp.CalculatorHydrodynamicProfile(configuration, 0)

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))
    prof.update()

print(sim.summary())

prof.read()

rp.tools.save_configuration(configuration, "final.xyz")
