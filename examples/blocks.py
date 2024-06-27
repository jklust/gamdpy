""" Example of a Simulation using rumdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import rumdpy as rp

# Setup configuration: FCC Lattice
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973)
configuration.randomize_velocities(T=0.8 * 2)

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, pairpot, integrator,
                    num_timeblocks=16, steps_per_timeblock=1024 * 2,
                    storage='LJ_T0.70.h5')

# Run simulation one block at a time
for block in sim.timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
