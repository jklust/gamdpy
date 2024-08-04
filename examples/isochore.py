""" Simple example of performing several simulation in one go using rumdpy.

Simulation of heating a Lennard-Jones crystal on an isochore in the NVT ensemble.
For an even simpler script, see minimal.py

"""

import rumdpy as rp

# Setup fcc configuration
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(T=2*0.8)

# Setup pair potential.
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

num_blocks = 8
steps_per_block = 1024*2

for temperature in ['0.70', '1.10', '1.50']:
    print('\n\nTemperature: ' + temperature)
    
    # Setup integrator
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup Simulation
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                        steps_between_momentum_reset=100,
                        storage='Data/LJ_r0.973_T'+temperature+'.h5') 

    print('Equilibration:')
    for block in sim.timeblocks():
        print(sim.status(per_particle=True))
    print(sim.summary())
    
    print('Production:')
    for block in sim.timeblocks():
        print(sim.status(per_particle=True))
    print(sim.summary())

# To get a plot of the MSD do something like this:
# python3 -m rumdpy.tools.calc_dynamics -o Data/msd_r0.973.pdf Data/LJ_r0.973_T*.h5
