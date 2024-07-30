
import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   
import rumdpy as rp

class Gravity():

    def __init__(self, *args, verbose=False):

        nargin = len(args)

        if nargin == 2:
            force_array = args[0]
            indices_array = args[1]

        elif nargin == 3:
            ptype = args[0] # e.g. 0
            force = args[1] # e.g. [1, 0, 0]
            conf = args[2] 

            force_array, indices_array = [], []
            counter = 0
            for n in range(conf.N):
                if conf.ptype[n]==ptype:
                    indices_array.append( [counter, n] )
                    force_array.append( force )
                    counter = counter + 1
        else:
            raise ValueError("Incorrect number of arguments to constructor")


        self.force_array = np.array(force_array, dtype=np.float32)
        self.indices_array = np.array(indices_array, dtype=np.int32) 

        if verbose:
            print(f"{self.force_array} \n {self.indices_array}")

    
    def get_params(self, configuration, compute_plan, verbose=False):

        self.d_pindices = cuda.to_device(self.indices_array)
        self.d_force = cuda.to_device(self.force_array);
        
        return (self.d_pindices, self.d_force)


    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1
    
        f_id = configuration.vectors.indices['f'] 
        
        def gravity_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            f = vectors[f_id][indices[1]]
            extforce = values[indices[0]][0]

            f[0] = f[0] + extforce

            return
    
        return make_fixed_interactions(configuration, gravity_calculator, compute_plan, verbose=False)
    


