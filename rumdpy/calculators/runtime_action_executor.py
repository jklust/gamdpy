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
    
    def momentum_sum():
        

        return

    def momentum_reset():

        return


    if gridsync:
        def kernel(grid, vectors, scalars, r_im, sim_box, time):

            momentum_sum()
            grid.sync()
            momentum_reset()
            return
        return cuda.jit(device=gridsync)(kernel)
    else:
        def kernel(grid, vectors, scalars, r_im, sim_box, time):
            momentum_sum[num_blocks, (pb, 1)]()
            momentum_reset[num_blocks, (pb, 1)]()
            return
        return kernel

