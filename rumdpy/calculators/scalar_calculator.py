import numpy as np
import numba
import math
from numba import cuda

def make_scalar_calculator(configuration, steps_between_output, compute_plan, verbose=False):
    D = configuration.D
    num_part = configuration.N
    print(compute_plan)
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    UtilizeNIII = compute_plan['UtilizeNIII']
    num_blocks = (num_part-1)//pb + 1
    
    # Unpack indicies for vectors and scalars    
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
 
    
    def scalar_calculator(grid, vectors, scalars, r_im, sim_box,  output_array, step):
        """     
        """

        if step%steps_between_output==0:
            save_index = step//steps_between_output
        
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block * pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_part and my_t == 0:
                cuda.atomic.add(output_array, (save_index, 0), scalars[global_id][u_id])   # Potential energy
                cuda.atomic.add(output_array, (save_index, 1), scalars[global_id][w_id])   # Virial
                cuda.atomic.add(output_array, (save_index, 2), scalars[global_id][lap_id]) # Laplace
                cuda.atomic.add(output_array, (save_index, 3), scalars[global_id][fsq_id]) # F**2
                cuda.atomic.add(output_array, (save_index, 4), scalars[global_id][k_id])   # Kinetic energy
        return
    return scalar_calculator