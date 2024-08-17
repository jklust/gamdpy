
import numba
import numpy as np
from numba import cuda

from .make_fixed_interactions import make_fixed_interactions  # tether is an example of 'fixed' interactions


class Tether:
    """ Connect particles to harmonic tethers anchored to points in space. """

    def __init__(self):
        anchor_points_set = False


    def set_anchor_points_from_lists(self, particle_indices, ksprings):
    
        nsprings, nparticles = len(ksprings), len(particle_indices)
        
        if nsprings != nparticles:
            raise ValueError("Each particle must have exactly one spring connection - array must be same length");

        indices, tether_params = [], []
        for n in range(nparticles):
            indices.append([n, pindices[n]])
            pos = configuration['r'][n]
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

        if self.anchor_points_set==False:
            raise ValueError("Anchor points not defined")

        self.d_pindices = cuda.to_device(self.indices)
        self.d_tether_params = cuda.to_device(self.tether_params);
        
        return (self.d_pindices, self.d_tether_params)


    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1
    
        # Get indices values (instead of dictonary entries) 
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        #u_id, w_id, lap_id, m_id = [configuration.sid[key] for key in ['u', 'w', 'lap', 'm']]

        dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
        
        def tether_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            dr = cuda.local.array(shape=D,dtype=numba.float32)
            dist_sq = dist_sq_dr_function(values[indices[0]][:D], vectors[r_id][indices[1]], sim_box, dr)
            
            spring = values[indices[0]][3]
            #u = numba.float32(0.5)*spring*dist_sq
            #s = -spring
            #umm = spring
            
            f=vectors[f_id][indices[1]];

            for k in range(D):
                f[k] = f[k] + dr[k]*spring
               # cuda.atomic.add(vectors, (f_id, indices[1], k), -dr[k]*s)      # Force
               # cuda.atomic.add(scalars, (indices[0], w_id), dr[k]*dr[k]*s*virial_factor)    # Virial
               # cuda.atomic.add(scalars, (indices[1], w_id), dr[k]*dr[k]*s*virial_factor)                      
        
            #cuda.atomic.add(scalars, (indices[1], u_id), u*numba.float32(0.5)) # Potential energy 
            #cuda.atomic.add(scalars, (indices[1], u_id), u*numba.float32(0.5))
            #lap = numba.float32(1-D)*s + umm                                   # Laplacian  
            #cuda.atomic.add(scalars, (indices[0], lap_id), lap)               
            #cuda.atomic.add(scalars, (indices[1], lap_id), lap)                
        
            return
    
        return make_fixed_interactions(configuration, tether_calculator, compute_plan, verbose=False)
    



    

