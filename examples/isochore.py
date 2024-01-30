""" Simple example of performing several simulation in one go using rumdpy.

Simulation of heating a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp
from rumdpy.integrators import nvt

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)

# Setup pair potential.
compute_plan = rp.get_default_compute_plan(configuration) # avoid
pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
params = [[[4.0, -4.0, 2.5], ], ]
pair_potential = rp.PairPotential(configuration, pairpot_func, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = pair_potential.get_interactions(configuration, exclusions=None, compute_plan=compute_plan, verbose=False) # move to Sim

num_blocks = 16
steps_per_block = 1024*2
temperatures = ['0.70', '1.10', '1.50', '1.90']

for temperature in temperatures:
    print('\n\nTemperature:' + temperature)
    
    # Setup integrator
    integrator = nvt.setup_new(configuration, temperature=float(temperature), tau=0.2, dt=0.005, 
                               compute_plan=compute_plan, verbose=False)

    # Setup Simulation
    sim = rp.Simulation_new(configuration, pairs, integrator, num_blocks, steps_per_block, 
                            compute_plan, include_rdf=False, 
                            storage='hdf5', filename='Data/LJ_r0.973_T'+temperature) 

    # Equilibrate
    for block in sim.run_blocks():
        pass
    
    # Production
    for block in sim.run_blocks():
        sim.print_status(per_particle=True)
    sim.print_summary()

# To get a plot of the MSD do something like this:
# python3 ../rumdpy/tools/calc_dynamics.py -o Data/msd_r0.973.pdf Data/LJ_r0.973_T*.h5