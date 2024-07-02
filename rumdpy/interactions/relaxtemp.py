
import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # tether is an example of 'fixed' interactions
import rumdpy as rp


class Relaxtemp():

    def __init__(self, relax_params, indices_array, verbose=False):

        self.relax_params = np.array(relax_params, dtype=np.float32)
        self.indices_array = np.array(indices_array, dtype=np.int32) 

        if self.relax_params.shape[0] != self.indices_array.shape[0]:
            raise ValueError("Length of indicies must be the same as length of relaxation parameters") 

        if verbose:
            print(f"{self.relax_params} \n {self.indices_array}")

    def get_params(self, configuration, compute_plan, verbose=False):

        self.d_pindices = cuda.to_device(self.indices_array)
        self.d_relax_params = cuda.to_device(self.relax_params);
        
        return (self.d_pindices, self.d_relax_params)


    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1
    
        # Get indices values (instead of dictonary entries) 
        v_id = configuration.vectors.indices['v'] 
        m_id = configuration.sid['m']
        
        def relaxtemp_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            v=vectors[v_id][indices[1]]
            m=scalars[indices[1]][m_id]
            Tdesired = values[indices[0]][0]
            tau = values[indices[0]][1]

            Tparticle = m/3.0*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

            fac = math.sqrt( 1.0 + tau*(Tdesired/Tparticle - 1.0) )

            for k in range(D):
                v[k] = v[k]*fac

            return
    
        return make_fixed_interactions(configuration, relaxtemp_calculator, compute_plan, verbose=False)
    



    

