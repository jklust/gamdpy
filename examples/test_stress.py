""" Example of a Simulation using rumdpy, including the stress tensor (configurational part only).

"""

import rumdpy as rp
import numpy as np

# Setup configuration: FCC Lattice
rho = 0.973
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=rho)
configuration.randomize_velocities(temperature=0.8 * 2)

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, pairpot, integrator,
                    num_timeblocks=16, steps_per_timeblock=1024 * 2,
                    steps_between_momentum_reset=100,
                    storage='LJ_T0.70.h5', compute_flags={'stresses':True})

print(sim.compute_plan)

# Run simulation one block at a time
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    vol = configuration.simbox.volume(configuration.simbox.lengths)
    
    sts_x_row = (np.sum(sim.configuration['sx'], axis=0)/2/vol)
    sts_y_row = (np.sum(sim.configuration['sy'], axis=0)/2/vol)
    sts_z_row = (np.sum(sim.configuration['sz'], axis=0)/2/vol)
    print('Stress tensor (configurational part)')
    print(sts_x_row)
    print(sts_y_row)
    print(sts_z_row)
    mean_diagonal = (sts_x_row[0] + sts_y_row[1] + sts_z_row[2])/3
    virial = np.mean(configuration['w'])
    print(f'Compare isotropic part with virial*rho: {mean_diagonal:.8f}  {virial*rho:.8f}' )

print(sim.summary())

