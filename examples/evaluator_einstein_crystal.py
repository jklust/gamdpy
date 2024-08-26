"""
Simulate Lennard-Jones crystal and evaluate the harmonic potential. 

This script sets up a configuration with a face-centered cubic (FCC) lattice 
and randomizes the velocities. The pair potential is the single component 12-6 
Lennard-Jones potential. The integrator is NVT. 
An evaluator is created to calculate the potential energy of the system 
using a none-interacting potential (eps=0.0) and harmonic springs. 
The simulation is then performed, and the mean potential energy of the reference
einstein crystal (harmonic springs) is calculated and printed.

"""

import numpy as np

import rumdpy as rp

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(T=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
eps, sig, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[eps, sig, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup Simulation.
sim = rp.Simulation(
    configuration, pair_pot, integrator,
    steps_between_momentum_reset=100,
    num_timeblocks=16,
    steps_per_timeblock=1024,
    scalar_output=32,
    storage='memory'
)

# Create evaluator for einstein crystal
#     (replace with your potential of interest)
none_interacting = pair_pot = rp.PairPotential(pair_func, params=[0.0, 1.0, 2.5], max_num_nbs=1000)
harmonic_springs = rp.Tether()
harmonic_springs.set_anchor_points_from_lists(
    particle_indices=list(range(configuration.N)),
    spring_constants=[1.0]*configuration.N,
    configuration=configuration
)
evaluator = rp.Evaluater(sim.configuration, [none_interacting, harmonic_springs])

# Run simulation
u_ipl = []
for block in sim.timeblocks():
    evaluator.evaluate(sim.configuration)
    u_ipl.append(np.sum(evaluator.configuration['u']))

print(f'Mean IPL potential energy: {np.mean(u_ipl)}')

