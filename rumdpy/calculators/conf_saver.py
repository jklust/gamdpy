import numpy as np
import numba
import math
from numba import cuda

def make_conf_saver(configuration, compute_plan, verbose=False):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    UtilizeNIII = compute_plan['UtilizeNIII']
    num_blocks = (num_part-1)//pb + 1
    
    # Unpack indices for vectors and scalars
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indices[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
 
    
    def conf_saver(grid, vectors, scalars, r_im, sim_box,  conf_array, step):
        """     
        """
        
        Flag = False
        if step==0:
            Flag = True
            save_index = 0
        else:
            b = np.int32(math.log2(np.float32(step)))
            c = 2**b
            if step==c:
                Flag = True
                save_index = b+1
            
        #my_block = cuda.blockIdx.x
        #local_id = cuda.threadIdx.x
        #global_id = my_block * pb + local_id
        #my_t = cuda.threadIdx.y
        #if global_id == 0 and my_t == 0:
        #    if Flag:
        #        print(step, save_index, 'True')
            #else:
            #    print(step, save_index, 'False')
        
        if Flag:
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block * pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_part and my_t == 0:
                for k in range(D):
                    conf_array[save_index, 0, global_id, k] = vectors[r_id][global_id,k]
                    conf_array[save_index, 1, global_id, k] = np.float32(r_im[global_id,k])
        return
    return conf_saver
