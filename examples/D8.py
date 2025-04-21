""" A simulation in eight dimensional space """

import numba

import gamdpy as rp

N = 1024  # Number of particles (65536)
temperature = 1.0

configuration = rp.Configuration(D=8)
configuration.make_positions(N, rho=1.0)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature)

pair_func = numba.njit(rp.harmonic_repulsion)
pair_potential = rp.PairPotential(pair_func, params=[2.0, 1.0], max_num_nbs=8192)

integrator = rp.integrators.NVT(temperature=temperature, tau=0.08, dt=0.001)

runtime_actions = [rp.MomentumReset(100), 
                   rp.ConfigurationSaver(), 
                   rp.ScalarSaver(32)]

sim = rp.Simulation(configuration, pair_potential, integrator, runtime_actions, 
                    num_timeblocks=16, steps_per_timeblock=128,
                    storage='memory')

for _ in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())
