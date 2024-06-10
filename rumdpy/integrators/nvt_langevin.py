""" 
NVT Langevin Leap-frog integrator
REF: https://arxiv.org/pdf/1303.7011.pdf
"""

import numpy as np
import numba
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from rumdpy.integrators.make_integrator import make_integrator, make_integrator_with_output

def _make_step_nvt_langevin(configuration, temperature_function, compute_plan, verbose=True):
    from numba.cuda.random import xoroshiro128p_normal_float32

    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating NVT langevin integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')

    # Unpack indices for vectors and scalars
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indices[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    temperature_function = numba.njit(temperature_function)
    apply_PBC_dimension = numba.njit(configuration.simbox.apply_PBC_dimension)
    # @cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync)
    # @cuda.jit(device=gridsync)
    def step_nvt_langevin(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NVT Langevin timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]
            REF: https://arxiv.org/pdf/1303.7011.pdf
        """

        dt, alpha, rng_states, old_beta = integrator_params
        temperature = temperature_function(time)

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        if global_id < num_part and my_t == 0:
            my_r = vectors[r_id][global_id]
            my_v = vectors[v_id][global_id]
            my_f = vectors[f_id][global_id]
            my_m = scalars[global_id][m_id]
            my_k = numba.float32(0.0)  # Kinetic energy
            my_fsq = numba.float32(0.0)  # force squared energy
            

            for k in range(D):
                # REF: https://arxiv.org/pdf/1303.7011.pdf sec. 2.C.
                random_number = xoroshiro128p_normal_float32(rng_states, global_id)
                beta = math.sqrt(numba.float32(2.0) * alpha * temperature * dt) * random_number
                # Eq. (16) in https://arxiv.org/pdf/1303.7011.pdf
                numerator =   numba.float32(2.0)*my_m - alpha * dt
                denominator = numba.float32(2.0)*my_m + alpha * dt
                a = numerator / denominator
                b_over_m = numba.float32(2.0) / denominator
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  # Half step kinetic energy
                my_fsq += my_f[k] * my_f[k]
                my_v[k] = a * my_v[k] + b_over_m * my_f[k] * dt + b_over_m * np.float32(0.5)*(beta+old_beta[global_id,k])
                old_beta[global_id,k] = beta # Store beta for next step
                my_r[k] += my_v[k] * dt

                apply_PBC_dimension(my_r, r_im[global_id], sim_box, k)

            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return

    if gridsync:
        return cuda.jit(device=gridsync)(step_nvt_langevin)  # return device function
    else:
        return cuda.jit(device=gridsync)(step_nvt_langevin)[
            num_blocks, (pb, 1)]  # return kernel, incl. launch parameters

def setup(configuration, interactions, temperature_function, alpha, dt, seed, compute_plan, verbose=True):
    """
    add setup docstring
    """
   
    integrator_step = _make_step_nvt_langevin(configuration, temperature_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan, verbose=verbose)
    
    rng_states = create_xoroshiro128p_states(configuration.N, seed=seed)
    old_beta = np.zeros((configuration.N, configuration.D), dtype=np.float32)
    d_old_beta = cuda.to_device(old_beta)
    integrator_params = (np.float32(dt), np.float32(alpha), rng_states, d_old_beta) # Needs to be compatible with unpacking in
                                                                                    # step_nvt_langevin()
    return integrate, integrator_params


def setup_output(configuration, interactions, output_calculator, temperature_function, alpha, dt, seed, compute_plan, verbose=True):
    
    integrator_step = _make_step_nvt_langevin(configuration, temperature_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator_with_output(configuration, integrator_step, interactions, output_calculator, compute_plan=compute_plan,
                                verbose=verbose)

    rng_states = create_xoroshiro128p_states(configuration.N, seed=seed)
    old_beta = np.zeros((configuration.N, configuration.D), dtype=np.float32)
    d_old_beta = cuda.to_device(old_beta)
    integrator_params = (np.float32(dt), np.float32(alpha), rng_states, d_old_beta) # Needs to be compatible with unpacking in
                                                                                    # step_nvt_langevin()
    return integrate, integrator_params
