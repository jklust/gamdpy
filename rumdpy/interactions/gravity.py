
import numpy as np
from numba import cuda
import numba
import math
from .make_fixed_interactions import make_fixed_interactions


class Gravity:
    """Adding a gravitational-like force on particles.

        Parameters
        ----------
        (a) A list of particle indices on which the forces act and associated list of forces 
        (b) A list of particle types one which the forces act and associated list of forces

        At the moment the force will act in the x-direction 

        See examples/poiseuille.py
    """ 
   
    def __init__(self):
       self.indices_set = False


    def set_gravity_from_lists(self, particle_indices, forces):

        if len(forces) != len(particle_indices):
            raise ValueError("Force and particle index arrays must have same length")

        force_array, indices_array = [], []

        for n in range(len(particle_indices)):
            indices_array.append( [n, pindices[n]] )
            force_array.append( forces[n] )

        self.force_array = np.array(force_array, dtype=np.float32)
        self.indices_array = np.array(indices_array, dtype=np.int32) 

        self.indices_set = True


    def set_gravity_from_types(self, particle_types, forces, configuration):

        ntypes, nforces = len(particle_types), len(forces)

        if ntypes != nforces:
            raise ValueError("Force and particle type arrays must have same length")

        force_array, indices_array = [], []

        counter = 0
        for n in range(configuration.N):
            for m in range(ntypes):
                if configuration.ptype[n]==particle_types[m]:
                    indices_array.append( [counter, n] )
                    force_array.append( forces[m] )
                    counter = counter + 1
                    break

        self.force_array = np.array(force_array, dtype=np.float32)
        self.indices_array = np.array(indices_array, dtype=np.int32) 

        self.indices_set = True


    def get_params(self, configuration, compute_plan, verbose=False):
        
        if self.indices_set == False:
            raise ValueError("Indices not defined")

        self.d_pindices = cuda.to_device(self.indices_array)
        self.d_force = cuda.to_device(self.force_array);
        
        return (self.d_pindices, self.d_force)


    def get_kernel(self, configuration, compute_plan, compute_flags, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1
    
        compute_u = compute_flags['U'] # PE should be included ?!?!
        # Note virial, lapacian, stresses are zero for gravity
        f_id = configuration.vectors.indices['f'] 
       
        def gravity_calculator(vectors, scalars, ptype, sim_box, indices, values):
       
            f = vectors[f_id][indices[1]]
            
            f[0] = f[0] + values[indices[0]]
            
            return
    
        return make_fixed_interactions(configuration, gravity_calculator, compute_plan, verbose=False)
    


