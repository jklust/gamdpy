import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions

##################################################################
#### Bonds
##################################################################

def make_bond_calculator(configuration, bondpotential_function):
    
    D = configuration.D
    dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

    # Unpack indicies for vectors and scalars    
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
        
    bondpotential_function = numba.njit(bondpotential_function)
    
    def bond_calculator(vectors, scalars, ptype, sim_box, indicies, values):
        
        dr = cuda.local.array(shape=D,dtype=numba.float32)
        dist_sq = dist_sq_dr_function(vectors[r_id][indicies[1]], vectors[r_id][indicies[0]], sim_box, dr)
        u, s, umm = bondpotential_function(math.sqrt(dist_sq), values[indicies[2]])
               
        for k in range(D):
            cuda.atomic.add(vectors, (f_id, indicies[0], k), -dr[k]*s)      # Force
            cuda.atomic.add(vectors, (f_id, indicies[1], k), +dr[k]*s)                      
            cuda.atomic.add(scalars, (indicies[0], w_id), dr[k]*dr[k]*s)    # Virial
            cuda.atomic.add(scalars, (indicies[1], w_id), dr[k]*dr[k]*s)                      
        cuda.atomic.add(scalars, (indicies[0], u_id), u*numba.float32(0.5)) # Potential enerrgy 
        cuda.atomic.add(scalars, (indicies[1], u_id), u*numba.float32(0.5))
        lap = numba.float32(1-D)*s + umm                                    # Laplacian  
        cuda.atomic.add(scalars, (indicies[0], lap_id), lap)               
        cuda.atomic.add(scalars, (indicies[1], lap_id), lap)                
        
        return
    return bond_calculator

def setup_bonds(configuration, bond_potential, potential_params_list, particles_list, compute_plan, verbose=True):
    D = configuration.D
    N = configuration.N
    num_types = len(potential_params_list)
    assert len(particles_list) == num_types

    total_number_indicies = 0 
    for particles in particles_list:
        total_number_indicies += particles.shape[0]
    
    if verbose:
        print(f'Setting up bond interactions: {N} particles, {num_types} bond types, {total_number_indicies} bonds in total.')

    bond_indicies = np.zeros((total_number_indicies, 3), dtype=np.int32)    
    bond_params = np.zeros((num_types, len(potential_params_list[0])), dtype=np.float32)
    
    start_index = 0  
    for bond_type in range(num_types):
        next_start_index = start_index + len(particles_list[bond_type])
       
        bond_indicies[start_index:next_start_index, 0:2] = particles_list[bond_type] 
        bond_indicies[start_index:next_start_index, 2] = bond_type 
        start_index = next_start_index
        
        bond_params[bond_type,:] = potential_params_list[bond_type] 
    
    bond_calculator =  make_bond_calculator(configuration, bond_potential)
    bond_interactions = make_fixed_interactions(configuration, bond_calculator, compute_plan, verbose=False)
    d_bond_indicies = cuda.to_device(bond_indicies)
    d_bond_params = cuda.to_device(bond_params)
    bond_interaction_params = (d_bond_indicies, d_bond_params)
    
    # Setup exclusion 'list'
    exclusions = np.zeros((N,10), dtype=np.int32)
    for particles in particles_list:
        add_exclusions_from_bond_indicies(exclusions, particles)
    d_exclusions = cuda.to_device(exclusions)
 
    return {'interactions': bond_interactions, 
            'interaction_params': bond_interaction_params, 
           'exclusions': d_exclusions}


def add_exclusions_from_bond_indicies(exclusions, bond_indicies):
    num_part, max_num_exclusions = exclusions.shape
    max_num_exclusions -= 1 # Last index used for number of exclusions for given particle
    
    for bond in range(bond_indicies.shape[0]):
        i = bond_indicies[bond,0]
        j = bond_indicies[bond,1]
        
        if exclusions[i,-1] < max_num_exclusions:
            exclusions[i,exclusions[i,-1]] = j
        exclusions[i,-1] += 1
        
        if exclusions[j,-1] < max_num_exclusions:
            exclusions[j,exclusions[j,-1]] = i
        exclusions[j,-1] += 1

    assert np.max(exclusions[:,-1]) <= max_num_exclusions
    
