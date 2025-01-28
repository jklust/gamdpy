""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp

rho = 0.8442

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
#configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
#configuration.make_lattice(rp.unit_cells.FCC, cells=[16, 16, 16], rho=rho)
configuration.make_lattice(rp.unit_cells.FCC, cells=[32, 32, 32], rho=rho)
#configuration.make_lattice(rp.unit_cells.FCC, cells=[64, 64, 64], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=1.44)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=500)

# Setup integrator:
dt = 0.005
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=dt)
integrator = rp.integrators.NVE(dt=dt)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [#rp.ConfigurationSaver(), 
                   #rp.ScalarSaver(32), 
                   rp.MomentumReset(100)]

compute_plan = rp.get_default_compute_plan(configuration)
#compute_plan['UtilizeNIII'] = False
#compute_plan['gridsync'] = True
#compute_plan['nblist'] = 'N squared'
#compute_plan['skin'] = 0.7
#compute_plan['pb'] = 32
#compute_plan['tp'] = 8

# Setup Simulation. 
sim = rp.Simulation(configuration, [pair_pot, ], integrator, runtime_actions, 
                    num_timeblocks=4, steps_per_timeblock=1*512,
                    storage='memory', compute_plan=compute_plan)

print('Initial compute_plan: ', sim.compute_plan)

# Run simulation
for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())

# Print current status of configuration
print(configuration)


sim.autotune_bruteforce(verbose=False)
print('Optimized compute_plan: ', sim.compute_plan)

# Run simulation
for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())

# Print current status of configuration
print(configuration)


# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
