""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)

# Setup pair potential.
pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
params = [[[4.0, -4.0, 2.5], ], ]
pair_potential = rp.PairPotential2(pairpot_func, params=params, max_num_nbs=1000)

# Setup integrator
integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
num_blocks = 16
steps_per_block = 1024*2
sim = rp.Simulation(configuration, pair_potential, integrator, 
                    num_blocks, steps_per_block, storage='LJ_T0.70.h5') 

# Run simulation at a time
for block in sim.blocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
