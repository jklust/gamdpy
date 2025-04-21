""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [rp.ConfigurationSaver(), 
                   rp.ScalarSaver(), 
                   rp.MomentumReset(100)]

# Setup Simulation. 
sim = rp.Simulation(configuration, [pair_pot, ], integrator, runtime_actions, 
                    num_timeblocks=32, steps_per_timeblock=1024,
                    storage='LJ_T0.70.h5')

# Run simulation
for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())

# Print current status of configuration
print(configuration)


# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
