
import numpy as np
from numba import cuda
import numba
import math
from .make_fixed_interactions import make_fixed_interactions


class Gravity:
    """ Gravity force on particles. """
    
    def __init__(self, force, configuration, pindices=None, ptype = None, verbose=False):

        force_array, indices_array = [], []

        if pindices == None:
            
            if len(force) != len(ptype):
                raise ValueError("Force and particle type arrays must have same length")

            counter = 0
            for n in range(configuration.N):
                if configuration.ptype[n]==ptype:
                    indices_array.append( [counter, n] )
                    force_array.append( force )
                    counter = counter + 1

        elif ptype == None:

            if len(force) != len(pindices):
                raise ValueError("Force and particle index arrays must have same length")

            for n in range(len(pindices)):
                indices_array.append( [n, pindices[n]] )
                force_array.append( force[n] )

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
    
        #f_id = configuration.vectors.indices['f'] 
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
       
        def gravity_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            f = vectors[f_id][indices[1]]
            r = vectors[r_id][indices[1]]
            
            z0 = numba.float32(3.2)
            z1 = numba.float32(15.8)
            L = z1 - z0

            n = numba.float32(2.0)
            hL = numba.float32(0.5)*z1

            amplitude = values[indices[0]][0]
            
            f[0] = f[0] + amplitude*math.sin( n*3.1416*(r[2] + hL - z0)/L )

            return
    
        return make_fixed_interactions(configuration, gravity_calculator, compute_plan, verbose=False)
    


