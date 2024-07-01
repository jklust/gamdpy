""" Example of a nanoslit pore simulation using rumdpy.

"""
import numpy as np
import random 
import rumdpy as rp

springconstant = 300.0

# Setup configuration: FCC Lattice
nx, ny, nz, rhoWall, rhoFluid = 12, 12, 20, 1.0, 0.8;
configuration = rp.make_configuration_fcc(nx, ny, nz, rhoWall)

# Fluid particles have type '0', wall particles '1', dummy particles '2'
nwall, npart = nx*ny*4, nx*ny*nz
configuration.ptype[:nwall-1] = 1
configuration.ptype[npart-nwall-1:] = 1
nfluid = np.sum( configuration.ptype==0 )
nfluidWanted = int(nfluid*rhoFluid/rhoWall)

while nfluid > nfluidWanted:
    idx = random.randint(nwall, npart-nwall)
    if configuration.ptype[idx] == 0:
        configuration.ptype[idx] = 2
        nfluid = nfluid - 1

# Tether the wall particles
particle_indices = []
tether_parameters = []
for n in range(npart):
    if configuration.ptype[n]==1:
        particle_indices.append(n)
        pos =  configuration['r'][n]
        tether_parameters.append( [pos[0], pos[1], pos[2], 300.0] )

#particle_indices = np.where(configuration.ptype==1)[0] #<- convert to list comprehension
#tether_parameters = np.concatenate( (configuration['r'][particle_indices], np.ones((nwall,1))*300 ), axis=1)
#springs = [np.ones( (nwall,1) )*springconstant] 

tether = rp.Tether(tether_parameters, particle_indices)

# Set the pair interactions
pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 0.0], 
       [1.0, 1.0, 0.0], 
       [0.0, 0.0, 0.0]] 
eps = [[1.0, 1.0, 0.0], 
       [1.0, 1.0, 0.0], 
       [0.0, 0.0, 0.0]]
cut = np.array(sig)*2.5 
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    # Temperature
configuration.randomize_velocities(T=0.8 * 2)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)
compute_plan = rp.get_default_compute_plan(configuration)
compute_plan['gridsync']=False
compute_plan['tp']=1 

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, [pairpot,tether], integrator,
                    num_timeblocks=16, steps_per_timeblock=1024 * 2,
                    storage='LJ_T0.70.h5', compute_plan=compute_plan)

print(sim.compute_plan)

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
