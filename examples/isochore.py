""" Simple example of performing several simulation in one go using rumdpy.

Simulation of heating a Lennard-Jones crystal on an isochore in the NVT ensemble.
For an even simpler script, see minimal.py

"""

import rumdpy as rp

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)

# Setup pair potential.
compute_plan = rp.get_default_compute_plan(configuration) # avoid
pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
params = [[[4.0, -4.0, 2.5], ], ]
pair_potential = rp.PairPotential(configuration, pairpot_func, exclusions=None, params=params, max_num_nbs=1000, compute_plan=compute_plan)

num_blocks = 16
steps_per_block = 1024*2

for temperature in ['0.70', '1.10', '1.50', '1.90']:
    print('\n\nTemperature: ' + temperature)
    
    # Setup integrator
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup Simulation
    sim = rp.Simulation(configuration, pair_potential, integrator, num_blocks, steps_per_block,
                        storage='Data/LJ_r0.973_T'+temperature+'.h5') 

    print('Equilibration:')
    for block in sim.run_blocks():
        sim.print_status(per_particle=True)
    sim.print_summary()
    
    print('Production:')
    for block in sim.run_blocks():
        sim.print_status(per_particle=True)
    sim.print_summary()

# To get a plot of the MSD do something like this:
# python3 -m rumdpy.tools.calc_dynamics -o Data/msd_r0.973.pdf Data/LJ_r0.973_T*.h5