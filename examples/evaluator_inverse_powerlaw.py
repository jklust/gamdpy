""" Simulate Lennard-Jones system and evaluate the inverse power law potential. """ 

import numpy as np

import rumdpy as rp

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(T=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
pair_pot = rp.PairPotential(pair_func, params=[1.0, 1.0, 2.5], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=32,
                    steps_per_timeblock=2048,
                    scalar_output=4,   # 2048/4 = 512 scalar data points per timeblock.
                    storage='memory')  # 64*512 = 32768 scalar data points in total.

# Create evaluator for the inverse power law potential (replace with your potential of interest)
pair_func_ref = rp.apply_shifted_potential_cutoff(rp.LJ_12_6)
ipl12 = rp.PairPotential(pair_func_ref, params=[4.0, 0.0, 2.5], max_num_nbs=1000)
evaluator = rp.Evaluater(sim.configuration, ipl12)

# Run simulation
u_lj = []
u_ipl = []
for block in sim.timeblocks():
    u_lj.append(sim.configuration.get_potential_energy())
    evaluator.evaluate()
    u_ipl.append(evaluator.configuration.get_potential_energy())

print(f'Mean LJ potential energy: {np.mean(u_lj)}')
print(f'Mean IPL potential energy: {np.mean(u_ipl)}')
print(f'Correlation: {np.corrcoef(u_lj, u_ipl)[0, 1]}')

