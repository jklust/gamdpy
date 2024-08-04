""" Example of a BCC lattice simulation with Lennard-Jones potential.

This example demonstrates how to set up a different lattice than the default FCC lattice.
Note more lattices are available, or you can define your own lattice.

"""

import rumdpy as rp

# Setup configuration. BCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(unit_cell=rp.unit_cells.BCC, cells=[8, 8, 8], rho=1.0)

# Setup masses and velocities
configuration['m'] = 1.0  # Set all masses to 1.0
configuration.randomize_velocities(T=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_steps=32*1024, storage='memory')

# Run simulation
sim.run()
