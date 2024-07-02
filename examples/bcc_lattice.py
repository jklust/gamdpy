import rumdpy as rp

# Setup configuration. BCC Lattice
cells = [8, 8, 8]
positions, box_vector = rp.tools.make_lattice(rp.unit_cells.BCC, cells, rho=1.0)
configuration = rp.Configuration()
configuration['r'] = positions
configuration.simbox = rp.Simbox(configuration.D, box_vector)

# Setup masses and velocities
configuration['m'] = 1.0  # Set all masses to 1.0
configuration.randomize_velocities(T=0.7 * 2)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential2(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_steps=32*1024, storage='memory')

# Run simulation
sim.run()
