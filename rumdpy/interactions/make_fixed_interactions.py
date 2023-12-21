import numpy as np
import numba
import math
from numba import cuda

def make_fixed_interactions(configuration, fixed_potential, compute_plan, verbose=True,):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    num_blocks = (num_part-1)//pb + 1    

    if verbose:
        print(f'Generating fixed interactions for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')      

    # Unpack indicies for vectors and scalars    
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
    
    # Prepare user-specified functions for inclusion in kernel(s)
    # NOTE: Include check they can be called with right parameters and returns the right number and type of parameters 
    
    potential_calculator = numba.njit(fixed_potential)

    @cuda.jit( device=gridsync )
    def fixed_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
        indicies, values = interaction_parameters
        num_interactions = indicies.shape[0]
        num_threads = num_blocks*pb*tp

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        my_t = cuda.threadIdx.y
        #global_id = (my_block*pb + local_id)*tp + my_t
        global_id = (my_block*pb + local_id) + my_t*cuda.blockDim.x*cuda.gridDim.x # Faster
        
        for index in range(global_id, num_interactions, num_threads):
            potential_calculator(vectors, scalars, ptype, sim_box, indicies[index], values)

        return
    return fixed_interactions

