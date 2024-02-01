""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp
from rumdpy.integrators import nvt

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=7, ny=7, nz=7, rho=0.973, T=0.8 * 2)

# Setup pair potential.
compute_plan = rp.get_default_compute_plan(configuration) # avoid
pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
params = [[[4.0, -4.0, 2.5], ], ]
pair_potential = rp.PairPotential(configuration, pairpot_func, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = pair_potential.get_interactions(configuration, exclusions=None, compute_plan=compute_plan, verbose=False) # move to Sim

# Setup integrator
integrator = nvt.setup_new(configuration, temperature=0.70, tau=0.2, dt=0.005, compute_plan=compute_plan, verbose=False)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
num_blocks = 16
steps_per_block = 1024*2
sim = rp.Simulation_new(configuration, pairs, integrator, num_blocks, steps_per_block, 
                        compute_plan, storage='LJ_T0.70.h5') 

# Run Simulation
for block in sim.run_blocks():
    sim.print_status(per_particle=True)
sim.print_summary()

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
