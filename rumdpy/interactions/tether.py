
import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # tether is an example of 'fixed' interactions
import rumdpy as rp


class Tether():

    def __init__(self, spring_constants, configuration, pindices=None, anchor_points=None, ptypes=None, verbose=False):

        indices, tether_params = [], []
         
        if pindices == None: # Using part. type settings
            
            ntypes, nsprings = len(ptypes), len(spring_constants)
            if ntypes != nsprings:
                raise ValueError("Each type must have exactly one spring connection - arrays must be same length")
            
            counter = 0
            for n in range(configuration.N):
                for m in range(ntypes):
                    if configuration.ptype[n]==ptypes[m]:
                        indices.append([counter, n]) 
                        pos =  configuration['r'][n]
                        tether_params.append( [pos[0], pos[1], pos[2], spring_constants[m]] )
                        counter = counter + 1
                        break
        
        elif ptypes == None: # Using part. indices settings
            
            nsprings, nparticles = len(spring_constants), len(pindices)
            if nsprings != nparticles:
                raise ValueError("Each particle must have exactly one spring connection - array must be same length");

            for n in range(nparticles):
                indices.append([n, pindices[n]])
                pos = configuration['r'][n]
                tether_params.append( [pos[0], pos[1], pos[2], spring_constants[n]] )
                
        else:

            raise ValueError("Incorrect instantiation of tether")


        self.tether_params = np.array(tether_params, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32) 
    
        if verbose:
            print(f"{self.tether_params} \n {self.indices}")



    def get_params(self, configuration, compute_plan, verbose=False):

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
    



    

