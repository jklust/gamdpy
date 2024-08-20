
import numba
import numpy as np
from numba import cuda

from .make_fixed_interactions import make_fixed_interactions  # tether is an example of 'fixed' interactions


class Tether:
    """ Connect particles to anchor-points in space with a harmonic spring force. 
        
        Parameters
        ----------
        Points and spring constants are defined using either 
        (a) A list of particle indices to be tethered and associated list of spring constants
        (b) A list of particle types to be tethered and associated list of spring constants

        See examples/tethered_particles.py
    """

    
    def __init__(self):
        self.anchor_points_set = False


    def set_anchor_points_from_lists(self, particle_indices, ksprings, configuration):
    
        nsprings, nparticles = len(ksprings), len(particle_indices)
        
        if nsprings != nparticles:
            raise ValueError("Each particle must have exactly one spring connection - array must be same length");

        indices, tether_params = [], []
        for n in range(nparticles):
            indices.append([n, particle_indices[n]])
            pos = configuration['r'][particle_indices[n]]
            tether_params.append( [pos[0], pos[1], pos[2], ksprings[n]] )
        
        self.tether_params = np.array(tether_params, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32) 
    
        self.anchor_points_set = True


    def set_anchor_points_from_types(self, particle_types, spring_constants, configuration):

        nsprings, ntypes = len(spring_constants), len(particle_types)

        if ntypes != nsprings:
            raise ValueError("Each type must have exactly one spring connection - arrays must be same length")
            
        indices, tether_params, counter = [], [], 0
        for n in range(configuration.N):
            for m in range(ntypes):
                if configuration.ptype[n]==particle_types[m]:
                    indices.append([counter, n]) 
                    pos =  configuration['r'][n]
                    tether_params.append( [pos[0], pos[1], pos[2], spring_constants[m]] )
                    counter = counter + 1
                    break
         
        self.tether_params = np.array(tether_params, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32) 
    
        self.anchor_points_set = True

    def get_params(self, configuration, compute_plan, verbose=False):

        if self.anchor_points_set == False:
            raise ValueError("Anchor points not defined")

        self.d_pindices = cuda.to_device(self.indices)
        self.d_tether_params = cuda.to_device(self.tether_params);
        
        return (self.d_pindices, self.d_tether_params)


    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1
    
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
        
        def tether_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            dr = cuda.local.array(shape=D,dtype=numba.float32)
            dist_sq = dist_sq_dr_function(values[indices[0]][:D], vectors[r_id][indices[1]], sim_box, dr)
            
            spring = values[indices[0]][3]
           
            f=vectors[f_id][indices[1]];

            for k in range(D):
                f[k] = f[k] + dr[k]*spring
       
            return
    
        return make_fixed_interactions(configuration, tether_calculator, compute_plan, verbose=False)
    



    

