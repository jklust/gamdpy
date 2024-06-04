import numpy as np
import numba
import math
from numba import cuda


def make_runtime_action_executor(configuration, steps_between_action, compute_plan, verbose=False):

    # Unpack parameters from configuration and compute_plan
    D, num_part = configuration.D, configuration.N
    pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
    num_blocks = (num_part - 1) // pb + 1

    # Unpack indices for vectors and scalars to be compiled into kernel
    v_id = configuration.vectors.indicies['v']
    m_id = configuration.sid['m']
    
    
    def zero_momentum(cm_velocity):
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y
        if global_id == 0 and my_t == 0:
            for k in range(D):
                cm_velocity[k] = np.float32(0.)
        return
    
    def sum_momentum(vectors, scalars, cm_velocity):
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y
        if my_t == 0:
            my_m = scalars[global_id][m_id]
            for k in range(D):
                cuda.atomic.add(cm_velocity, k, my_m * vectors[v_id][global_id][k])
            cuda.atomic.add(cm_velocity, D, my_m) # Total mass summed in last index of cm_velocity
        return

    def shift_velocities(vectors, cm_velocity):
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y
        if my_t == 0:
            for k in range(D):
                vectors[v_id][global_id,k] -= cm_velocity[k] / cm_velocity[D] 


        return

    zero_momentum = cuda.jit(device=gridsync)(zero_momentum)
    sum_momentum = cuda.jit(device=gridsync)(sum_momentum)
    shift_velocities = cuda.jit(device=gridsync)(shift_velocities)

    if gridsync:
        def kernel(grid, vectors, scalars, r_im, sim_box, time, cm_velocity):
            zero_momentum(cm_velocity)
            grid.sync()
            sum_momentum(vectors, scalars, cm_velocity)
            grid.sync()
            shift_velocities(vectors, cm_velocity)
            return
        return cuda.jit(device=gridsync)(kernel)
    else:
        def kernel(grid, vectors, scalars, r_im, sim_box, time, cm_velocity):
            zero_momentum[1, 1](cm_velocity)
            sum_momentum[num_blocks, (pb, 1)](vectors, cm_velocity)
            shift_velocities[num_blocks, (pb, 1)](vectors, cm_velocity)
            return
        return kernel

