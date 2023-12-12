import numpy as np
import numba
from numba import cuda
import math
from rumdpy.integrators.make_integrator import make_integrator


def make_step_nve_toxvaerd(configuration, compute_plan, verbose=True):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating NVE integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')

    # Unpack indicies for vectors and scalars
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    apply_PBC_dimension = numba.njit(configuration.simbox.apply_PBC_dimension)

    # @cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync)
    # @cuda.jit(device=gridsync)
    def step_nve_toxvaerd(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NVE timestep using Leap-frog and the Toxvaerd scheme for the kinetic energy
            Kernel configuration: [num_blocks, (pb, tp)]        
        """

        dt, = integrator_params

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
                my_fsq += my_f[k] * my_f[k]
                v_squared = numba.float32(0.0)
                v_squared += numba.float32(0.5) * my_v[k] * my_v[k]
                v_squared -= numba.float32(0.25) * my_f[k] * my_f[k] / (my_m * my_m) * dt * dt
                my_v[k] += my_f[k] / my_m * dt
                v_squared += numba.float32(0.25) * my_v[k] * my_v[k]
                my_k += numba.float32(0.5) * my_m * v_squared
                my_r[k] += my_v[k] * dt

                apply_PBC_dimension(my_r, r_im[global_id], sim_box, k)
            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return

    if gridsync:
        return cuda.jit(device=gridsync)(step_nve_toxvaerd)  # return device function
    else:
        return cuda.jit(device=gridsync)(step_nve_toxvaerd)[
            num_blocks, (pb, 1)]  # return kernel, incl. launch parameters


def setup(configuration, interactions, dt, compute_plan, verbose=True):
    integrator_step = make_step_nve_toxvaerd(configuration, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan,
                                verbose=verbose)
    integrator_params = (np.float32(dt),)  # Needs to be compatible with unpacking in step_nve()

    return integrate, integrator_params


