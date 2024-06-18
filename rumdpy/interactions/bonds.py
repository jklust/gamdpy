import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions

class Bonds():
    def __init__(self, bond_potential, potential_params_list, particles_list):
        self.bond_potential = bond_potential
        self.potential_params_list = potential_params_list
        self.particles_list = particles_list

    def get_params(self, configuration, compute_plan, verbose=True):
        self.N = configuration.N
        self.num_types = len(self.potential_params_list)
        assert len(self.particles_list) == self.num_types

        self.total_number_indices = 0
        for particles in self.particles_list:
            self.total_number_indices += particles.shape[0]
    
        if verbose:
            print(f'Setting up bond interactions: {self.N} particles,')
            print(f'{self.num_types} bond types, {self.total_number_indices} bonds in total.')

        self.bond_indices = np.zeros((self.total_number_indices, 3), dtype=np.int32)
        self.bond_params = np.zeros((self.num_types, len(self.potential_params_list[0])), dtype=np.float32)
    
        start_index = 0  
        for bond_type in range(self.num_types):
            next_start_index = start_index + len(self.particles_list[bond_type])
            self.bond_indices[start_index:next_start_index, 0:2] = self.particles_list[bond_type]
            self.bond_indices[start_index:next_start_index, 2] = bond_type
            start_index = next_start_index
            self.bond_params[bond_type,:] = self.potential_params_list[bond_type]

        self.d_bond_indices = cuda.to_device(self.bond_indices)
        self.d_bond_params = cuda.to_device(self.bond_params)
        return (self.d_bond_indices, self.d_bond_params)
        
    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1
        assert compute_stresses == False # For now...

        if verbose:
            print('get_kernel: Bonds:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
            if compute_stresses:
                print('\tIncluding computation of stress tensor')
    
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        u_id, w_id, lap_id, m_id = [configuration.sid[key] for key in ['u', 'w', 'lap', 'm']]

        dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
        bondpotential_function = numba.njit(self.bond_potential)
    
        def bond_calculator(vectors, scalars, ptype, sim_box, indices, values):
            
            dr = cuda.local.array(shape=D,dtype=numba.float32)
            dist_sq = dist_sq_dr_function(vectors[r_id][indices[1]], vectors[r_id][indices[0]], sim_box, dr)
            u, s, umm = bondpotential_function(math.sqrt(dist_sq), values[indices[2]])
                
            for k in range(D):
                cuda.atomic.add(vectors, (f_id, indices[0], k), -dr[k]*s)      # Force
                cuda.atomic.add(vectors, (f_id, indices[1], k), +dr[k]*s)                      
                cuda.atomic.add(scalars, (indices[0], w_id), dr[k]*dr[k]*s)    # Virial
                cuda.atomic.add(scalars, (indices[1], w_id), dr[k]*dr[k]*s)                      
            cuda.atomic.add(scalars, (indices[0], u_id), u*numba.float32(0.5)) # Potential enerrgy 
            cuda.atomic.add(scalars, (indices[1], u_id), u*numba.float32(0.5))
            lap = numba.float32(1-D)*s + umm                                   # Laplacian  
            cuda.atomic.add(scalars, (indices[0], lap_id), lap)               
            cuda.atomic.add(scalars, (indices[1], lap_id), lap)                
            
            return
        
        return make_fixed_interactions(configuration, bond_calculator, compute_plan, verbose=False)

    def add_exclusions_from_bond_indices(exclusions, bond_indices):
        num_part, max_num_exclusions = exclusions.shape
        max_num_exclusions -= 1 # Last index used for number of exclusions for given particle
        
        for bond in range(bond_indices.shape[0]):
            i = bond_indices[bond,0]
            j = bond_indices[bond,1]
            
            if exclusions[i,-1] < max_num_exclusions:
                exclusions[i,exclusions[i,-1]] = j
            exclusions[i,-1] += 1
            
            if exclusions[j,-1] < max_num_exclusions:
                exclusions[j,exclusions[j,-1]] = i
            exclusions[j,-1] += 1

        assert np.max(exclusions[:,-1]) <= max_num_exclusions

    def get_exclusions(self):
        self.exclusions = np.zeros((self.N, 20), dtype=np.int32)
        for particles in self.particles_list:
            self.add_exclusions_from_bond_indices(self.exclusions, particles)
        return self.exclusions
    