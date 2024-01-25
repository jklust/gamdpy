""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import numpy as np

import rumdpy as rp
from rumdpy.integrators import nvt

# Setup fcc configuration
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)
configuration.copy_to_device()

# Setup pair potential.
compute_plan = rp.get_default_compute_plan(configuration)
pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
params = [[[4.0, -4.0, 2.5], ], ]
pair_potential = rp.PairPotential(configuration, pairpot_func, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = pair_potential.get_interactions(configuration, exclusions=None, compute_plan=compute_plan, verbose=False)

# Setup integrator
dt = np.float32(0.005)
temperature = rp.make_function_constant(value=0.7)
integrate, integrator_params = nvt.setup(configuration, pairs['interactions'], temperature,
                                         tau=0.2, dt=dt, compute_plan=compute_plan, verbose=False)


# Run Simulation
outer_steps = 10
inner_steps = 100
for outer_step in range(outer_steps):
    integrate(configuration.d_vectors, configuration.d_scalars, configuration.d_ptype, configuration.d_r_im, configuration.simbox.d_data,
              pairs['interaction_params'], integrator_params, np.float32(0.0), inner_steps)
    scalars = np.sum(configuration.d_scalars.copy_to_host(), axis=0)
    time = outer_step * inner_steps * dt
    print(f'\n{time= :<6.3}', end=' ')

    for name in configuration.sid:
        idx = configuration.sid[name]
        print(f'{name}= {scalars[idx]:<10.1f}', end=' ')
