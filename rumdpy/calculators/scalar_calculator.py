import numpy as np
import numba
import math
from numba import cuda

def make_scalar_calculator(configuration, steps_between_output, compute_plan, verbose=False):
    # Unpack parameters from configuration and compute_plan
    D, num_part = configuration.D, configuration.N
    pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
    num_blocks = (num_part - 1) // pb + 1
    
    # Unpack indices for scalars to be compiled into kernel  
    u_id, k_id, w_id, fsq_id, lap_id = [configuration.sid[key] for key in ['u', 'k', 'w', 'fsq', 'lap']]     
    v_id = configuration.vectors.indices['v']
    
    volume_function = numba.njit(configuration.simbox.volume)

    def scalar_calculator(grid, vectors, scalars, r_im, sim_box,  output_array, step):
        """     
        """

        if step%steps_between_output==0:
            save_index = step//steps_between_output
        
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                cuda.atomic.add(output_array, (save_index, 0), scalars[global_id][u_id])   # Potential energy
                cuda.atomic.add(output_array, (save_index, 1), scalars[global_id][w_id])   # Virial
                cuda.atomic.add(output_array, (save_index, 2), scalars[global_id][lap_id]) # Laplace
                cuda.atomic.add(output_array, (save_index, 3), scalars[global_id][fsq_id]) # F**2
                cuda.atomic.add(output_array, (save_index, 4), scalars[global_id][k_id])   # Kinetic energy
            

                cuda.atomic.add(output_array, (save_index, 6), vectors[v_id][global_id][0]) 
                cuda.atomic.add(output_array, (save_index, 7), vectors[v_id][global_id][1]) 
                cuda.atomic.add(output_array, (save_index, 8), vectors[v_id][global_id][2]) 

            if global_id == 0 and my_t == 0:
                output_array[save_index][5] = volume_function(sim_box)



                

        return
    return scalar_calculator

def extract_scalars(data, column_list, first_block=0):
    # Indcies hardcoded for now (see scalar_calculator above)

    column_indices = {'U':0, 'W':1, 'lapU':2, 'Fsq':3, 'K':4, 'Vol':5, 'vCMx':6, 'vCMy':7, 'vCMz':8}

    output_list = []
    for column in column_list:
        output_list.append(data['scalars'][first_block:,:,column_indices[column]].flatten())
    return tuple(output_list)
