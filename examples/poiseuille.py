""" Example of a nanoslit pore simulation using tethered LJ particles 

    The particles are tethered with a Hooke spring. The wall particles interact with a relaxation device.   
    All particles are integrated forward in time with the NVE integrator

    Wall density is set to 1.0 and fluid density to 0.8 - this is achieved by
    inclusion of a dummy particle type 

    Initial and final configurations are saved in xyz format for easy inspection in vmd
"""

import random

import numpy as np

import rumdpy as rp

# Setup a default fcc configuration
#nx, ny, nz = 6, 6, 10
#rhoWall = 1.0
#rhoFluid = 0.7
#configuration = rp.Configuration(D=3)
#configuration.make_lattice(rp.unit_cells.FCC, cells=[nx, ny, nz], rho=rhoFluid)
#configuration['m'] = 1.0
nxUnits, nyUnits, nzUnits, rhoWall, rhoFluid = 6, 6, 10, 1.0, 0.7;
configuration = rp.make_configuration_fcc(nxUnits, nyUnits, nzUnits, rhoWall)


# Fluid particles have type '0', wall particles '1', dummy particles '2'
nwall, npart = 0, configuration.N
hlz = 0.5 * configuration.simbox.lengths[2]
for n in range(npart):
    if configuration['r'][n][2] + hlz < 3.0:
        configuration.ptype[n] = 1
        nwall = nwall + 1

nfluid = np.sum(configuration.ptype == 0)
nfluidWanted = int(nfluid * rhoFluid / rhoWall)

while nfluid > nfluidWanted:
    idx = random.randint(0, npart - 1)
    if configuration.ptype[idx] == 0:
        configuration.ptype[idx] = 2
        nfluid = nfluid - 1

rp.tools.save_configuration(configuration, "initial.xyz")

# Tether specifications. 
# Alternative instantiation tether=rp.Tether(<index array>, <tether params>, verbose=False)
# where  index array: [row index in param, particle/atom index], tether parameters: [x0, y0, z0, kspring] 
tether = rp.Tether(ptypes=[1], spring_constants=[300.0], configuration=configuration)

# Add gravity force 
# Alternative instantiation grav = rp.Gravity(<index array>, <force>, verbose=False)
# where index array: [row index in param, particle/atom index], force:  scalar force
# Currently the force can only act in the x-direction
grav = rp.Gravity(ptype=[0], force=[0.01], configuration=configuration)

# Temp relaxation for wall particles
relax = rp.Relaxtemp(ptypes=[1], tau=[0.01], temperature=[2.0], configuration=configuration)

# Set the pair interactions
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
eps = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
cut = np.array(sig) * 2.5
pair_pot = rp.PairPotential2(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Temperature
configuration.randomize_velocities(T=2.0)

# Setup integrator: NVT
integrator = rp.integrators.NVE(dt=0.005)

# Compute plan
compute_plan = rp.get_default_compute_plan(configuration)
# Some compute plan settings for old cards 
#compute_plan['gridsync']=True
#compute_plan['tp']=6 

# Setup Simulation. Total number of time steps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, [pair_pot, tether, grav, relax], integrator,
                    num_timeblocks=100, steps_per_timeblock=64,
                    steps_between_momentum_reset=0, storage='memory', compute_plan=compute_plan)

prof = rp.CalculatorHydrodynamicProfile(configuration, 0)

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))
    prof.update()

print(sim.summary())

prof.read()

rp.tools.save_configuration(configuration, "final.xyz")
