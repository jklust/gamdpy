import numpy as np
import numba
import math
from numba import cuda


class PairPotential():
    """ Pair potential """

    def __init__(self, configuration, pairpotential_function, params, max_num_nbs, compute_plan):
        N = configuration.N
        D = configuration.D
        
        virial_factor = numba.float32( 0.5/D )
        def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, f, other_id):
            u, s, umm = pairpotential_function(ij_dist, ij_params)
            for k in range(D):
                my_f[k] = my_f[k] - dr[k]*s                         # Force
                cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
            cscalars[u_id] += u*numba.float32( 0.5 )                # Potential energy
            cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
            return

        virial_factor_NIII = numba.float32( 1.0/D )
        def pairpotential_calculator_NIII(ij_dist, ij_params, dr, my_f, cscalars, f, other_id):
            u, s, umm = pairpotential_function(ij_dist, ij_params)
            for k in range(D):
                cuda.atomic.add(f, (other_id, k), dr[k]*s)
                my_f[k] = my_f[k] - dr[k]*s                         # Force
                cscalars[w_id] += dr[k]*dr[k]*s*virial_factor_NIII  # Virial
            cscalars[u_id] += u                                      # Potential energy
            cscalars[lap_id] += (numba.float32(1-D)*s + umm)*numba.float32( 2.0 ) # Laplacian 
            return

        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]            # default: read from params array
            return result

        self.pairpotential_function = pairpotential_function
        if compute_plan['UtilizeNIII']:
            self.pairpotential_calculator = pairpotential_calculator_NIII
        else:
            self.pairpotential_calculator = pairpotential_calculator
        self.params_function = params_function
        self.params = params
        self.nblist = NbList(N, max_num_nbs)
        return
    
    def copy_to_device(self):
        self.d_params = cuda.to_device(self.params)
        self.nblist.copy_to_device()
    

# Helper functions

def make_potential_function_from_sympy(ufunc, param_names): 
    dufunc = sp.simplify(sp.diff(ufunc, r))                         # Sympy functions
    sfunc = sp.simplify(-sp.diff(ufunc, r)/r)
    ummfunc = sp.simplify(sp.diff(dufunc, r))
    u_lam = numba.njit(lambdify([r, param_names], ufunc, 'numpy'))  # Jitted python functions
    s_lam = numba.njit(lambdify([r, param_names], sfunc, 'numpy'))
    umm_lam = numba.njit(lambdify([r, param_names], ummfunc, 'numpy'))
   
    #@numba.njit
    def potential_function(r, params):
        u = np.float32(u_lam(r, params))
        s = np.float32(s_lam(r, params))
        umm =  np.float32(umm_lam(r, params))
        return u, s, umm

    return potential_function

def apply_shifted_potential_cutoff(pairpotential):  
    """
        Input:
            pairpotential: a function that calculates a pair-potential
                            u, s,  umm =  pairpotential(dist, params)
        Returns:
            potential: a function where shifted_potential_cutoff is applied to original function
                        (calls original potential function twice, avoiding changes to params)
    """ 
    pairpotential = numba.njit(pairpotential)
    @numba.njit
    def potential(dist, params):
        cut = params[-1]
        u,     s,     umm =     pairpotential(dist, params)
        u_cut, s_cut, umm_cut = pairpotential(cut,  params)
        u -= u_cut
        return u, s, umm
    return potential

def apply_shifted_force_cutoff(pairpotential):  # Cut-off by computing potential twice, avoiding changes to params
    """
        Input:
            pairpotential: a function that calculates a pair-potential
                            u, s,  umm =  pairpotential(dist, params)
        Returns:
            potential: a function where shifted force cutoff is applied to original function
                        (calls original potential function  twice, avoiding changes to params)
    """ 
    pairpotential = numba.njit(pairpotential)
    @numba.njit
    def potential(dist, params):
        cut = params[-1]
        u,     s,     umm =     pairpotential(dist, params)
        u_cut, s_cut, umm_cut = pairpotential(cut,  params)
        u -= u_cut - s_cut*dist*(dist-cut) 
        s -= s_cut
        return u, s, umm
    return potential

####################################################
### NBlist
####################################################'

class NbList():
    def __init__(self, num_part, max_num_nbs):
        self.nblist = np.zeros((num_part, max_num_nbs+1), dtype=np.int32)
        self.nbflag = np.zeros(3, dtype=np.int32)

    def copy_to_device(self):
        self.d_nblist = cuda.to_device(self.nblist)
        self.d_nbflag = cuda.to_device(self.nbflag)
    

####################################################
#### Interactions 
###################################################

def make_interactions(configuration, pair_potential, num_cscalars, compute_plan, verbose=True,):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    UtilizeNIII = compute_plan['UtilizeNIII']
    num_blocks = (num_part-1)//pb + 1    

    if verbose:
        print(f'Generating interactions for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')      

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
    
    # Prepare user-specified functions for inclusion in kernel(s)
    # NOTE: Include check they can be called with right parameters and returns the right number and type of parameters 
    ptype_function = numba.njit(configuration.ptype_function)
    params_function = numba.njit(pair_potential.params_function)
    pairpotential_calculator = numba.njit(pair_potential.pairpotential_calculator)
    dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

    #@cuda.jit('void(float32[:,:,:], float32[:], float32, int32[:])', device=gridsync)
    @cuda.jit( device=gridsync)
    def nblist_check(vectors, sim_box, skin, nbflag):
        """ Check validity of nblist, i.e. did any particle mode more than skin/2 since last nblist update?
            Each tread-block checks the assigned particles (global_id)
            nbflag[0] = 0          : No update needed
            nbflag[0] = num_blocks : Update needed
            Kernel configuration: [num_blocks, (pb, tp)]
        """

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y

        if nbflag[0]>0: # nblist update can be forced by setting nbflag[0]>0
            if global_id==0 and my_t==0:
                nbflag[0]=num_blocks
            #cuda.syncthreads()
        else:
            if global_id < num_part and my_t==0:
                dist_sq = dist_sq_function(vectors[r_id][global_id], vectors[r_ref_id][global_id], sim_box)
                #if numba.float32(4.)*dist_sq > skin*skin:
                if dist_sq > skin*skin*numba.float32(0.25):
                    nbflag[0]=num_blocks

        if global_id < num_part and my_t==0: # Initializion of forces moved here to make NewtonIII possible 
            for k in range(D):
                vectors[f_id][global_id, k] = numba.float32(0.0)
        return    

   
    #@cuda.jit('void(float32[:,:,:], float32[:], float32, int32[:], int32[:,:])', device=gridsync)
    @cuda.jit(device=gridsync)
    def nblist_update(vectors, sim_box, cut_plus_skin, nbflag, nblist, exclusions ):
        """ N^2 Update neighbor-list using numba.cuda 
            Kernel configuration: [num_blocks, (pb, tp)]
        """

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y

        if nbflag[0] > 0:
            max_nbs = nblist.shape[1]-1 # Last index is used for storing number of neighbors

            if global_id < num_part and my_t==0:
                nblist[global_id, max_nbs] = 0  # Set number of neighbors (stored at index max_nbs) to zero
                
            cuda.syncthreads() # nblist[global_id, max_nbs] ready
            
            if global_id < num_part:
                my_num_exclusions = exclusions[global_id,-1]
                for i in range(0, num_part, pb*tp): # Loop over blocks
                    for j in range(pb):             # Loop over particles the pb in block
                        other_global_id = j + i + my_t*pb   
                        if UtilizeNIII:
                            TwodN = 2*(other_global_id - global_id)
                            flag = other_global_id < num_part and (0 < TwodN <= num_part or TwodN < -num_part)
                        else:
                            flag = other_global_id != global_id and other_global_id < num_part
                        if flag:  
                            dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id], sim_box)
                            if dist_sq < cut_plus_skin*cut_plus_skin:
                                not_excluded = True  # Check exclusion list
                                for k in range(my_num_exclusions):
                                    if exclusions[global_id, k] ==  other_global_id:
                                        not_excluded = False
                                if not_excluded:
                                    my_num_nbs = cuda.atomic.add(nblist, (global_id, max_nbs), 1)   # Find next free index in nblist
                                    if my_num_nbs < max_nbs:                         
                                        nblist[global_id, my_num_nbs] = other_global_id     # Last entry is number of neighbors

            # Various house-keeping
            if global_id < num_part and my_t==0:
                for k in range(D):    
                    vectors[r_ref_id][global_id, k] = vectors[r_id][global_id, k]   # Store positions for wich nblist was updated ( used in nblist_check() ) 
            if local_id == 0 and my_t==0:
                cuda.atomic.add(nbflag, 0, -1)              # nbflag[0] = 0 by when all blocks are done
            if global_id == 0 and my_t==0:
                cuda.atomic.add(nbflag, 2, 1)               # Count how many updates are done in nbflag[2]
            if my_num_nbs >= max_nbs:                       # Overflow detected, nbflag[1] should be checked later, and then
                cuda.atomic.max(nbflag, 1, my_num_nbs)      # re-allocate larger nb-list, and redo computations from last safe state
 
        return 

    
    #@cuda.jit('void(float32[:,:,:], float32[:,:], int32[:], float32[:], int32[:,:], float32[:,:,:])', device=gridsync)  
    @cuda.jit( device=gridsync )  
    def calc_forces(vectors, cscalars, ptype, sim_box, nblist, params):
        """ Calculate forces as given by pairpotential_calculator() (needs to exist in outer-scope) using nblist 
            Kernel configuration: [num_blocks, (pb, tp)]        
        """
        
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y
        
        max_nbs = nblist.shape[1]-1
        
        if global_id < num_part and my_t==0: # Initialize global variables. Should be controlled by flag if more pair-potentials
            for k in range(num_cscalars):
                cscalars[global_id, k] = numba.float32(0.0)
            
        my_f = cuda.local.array(shape=D,dtype=numba.float32)
        my_dr = cuda.local.array(shape=D,dtype=numba.float32)
        my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)
       
        if global_id < num_part:
            for k in range(D):
                #my_r[k] = r[global_id, k]
                my_f[k] = numba.float32(0.0)
            for k in range(num_cscalars):
                my_cscalars[k] = numba.float32(0.0)
            my_type = ptype_function(global_id, ptype)
        
        cuda.syncthreads() # Make sure initializing global variables to zero is done

        if global_id < num_part:
            for i in range(my_t, nblist[global_id, max_nbs], tp):
                other_id = nblist[global_id, i] 
                other_type = ptype_function(other_id, ptype)
                dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                ij_params = params_function(my_type, other_type, params)
                cut = ij_params[-1]
                if dist_sq < cut*cut:
                    pairpotential_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, vectors[f_id], other_id)
            for k in range(D):
                cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
            for k in range(num_cscalars):
                cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

        return 

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            params, max_cut, skin, nblist, nbflag, exclusions = interaction_parameters
            #g = cuda.cg.this_grid() # Slow to do everytimestep, so added to parameters
            nblist_check(vectors, sim_box, skin, nbflag)
            grid.sync()
            nblist_update(vectors, sim_box, max_cut+skin, nbflag, nblist, exclusions)
            #g.sync() #not needed: same thread-block does nblist_update and calc_forces 
            calc_forces(vectors, scalars, ptype, sim_box, nblist, params)
            return
        return compute_interactions

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            params, max_cut, skin, nblist, nbflag, exclusions = interaction_parameters
            nblist_check[num_blocks, (pb, 1)](vectors, sim_box, skin, nbflag)
            nblist_update[num_blocks, (pb, tp)](vectors, sim_box, max_cut+skin, nbflag, nblist, exclusions)
            calc_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params)
            return
        return compute_interactions

def add_interactions(configuration, interactions0,  interactions1, compute_plan, verbose=True,):
    gridsync = compute_plan['gridsync']

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            interactions0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            grid.sync() # Not always necesarry !!!
            interactions1(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            return
        return compute_interactions

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            interactions0(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            interactions1(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            return
        return compute_interactions

def add_interactions_list(configuration, interactions_list, interaction_params_list, compute_plan, verbose=True,):
    gridsync = compute_plan['gridsync']
    length = len(interactions_list)
    
    # Unpack interaction functions from list/tuple. Would be nice to find more elegant way!!!
    i0 = interactions_list[0]
    i1 = interactions_list[1]
    if len(interactions_list)>2:
        i2 = interactions_list[2]
    if len(interactions_list)>3:
        i3 = interactions_list[3]
    if len(interactions_list)>4:
        i4 = interactions_list[4]
 
    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            grid.sync() # Not always necesarry !!!
            i1(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if length>2:
                grid.sync() # Not always necesarry !!!
                i2(grid, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if length>3:
                grid.sync() # Not always necesarry !!!
                i3(grid, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if length>4:
                grid.sync() # Not always necesarry !!!
                i4(grid, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return compute_interactions, tuple(interaction_params_list)

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            i1(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if length>2:
                i2(0, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if length>3:
                i3(0, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if length>4:
                i4(0, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return compute_interactions, tuple(interaction_params_list)

def make_fixed_interactions(configuration, fixed_potential, compute_plan, verbose=True,):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    num_blocks = (num_part-1)//pb + 1    

    if verbose:
        print(f'Generating fixed interactions for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')      

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
    
    # Prepare user-specified functions for inclusion in kernel(s)
    # NOTE: Include check they can be called with right parameters and returns the right number and type of parameters 
    
    potential_calculator = numba.njit(fixed_potential)

    @cuda.jit( device=gridsync )
    def fixed_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
        indicies, values = interaction_parameters
        num_interactions = indicies.shape[0]
        num_threads = num_blocks*pb*tp

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        my_t = cuda.threadIdx.y
        #global_id = (my_block*pb + local_id)*tp + my_t
        global_id = (my_block*pb + local_id) + my_t*cuda.blockDim.x*cuda.gridDim.x # Faster
        
        for index in range(global_id, num_interactions, num_threads):
            potential_calculator(vectors, scalars, ptype, sim_box, indicies[index], values)

        return
    return fixed_interactions

def make_bond_calculator(configuration, bondpotential_function):
    
    D = configuration.D
    dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
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
    num_types = len(potential_params_list)
    assert len(particles_list) == num_types

    total_number_indicies = 0 
    for particles in particles_list:
        total_number_indicies += particles.shape[0]
    print(total_number_indicies)

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
    bond_interactions = make_fixed_interactions(configuration, bond_calculator, compute_plan, verbose=verbose)
    d_bond_indicies = cuda.to_device(bond_indicies)
    d_bond_params = cuda.to_device(bond_params)
    bond_interaction_params = (d_bond_indicies, d_bond_params)

    return bond_interactions, bond_interaction_params


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


def make_planar_calculator(configuration, potential_function):
    
    D = configuration.D
    dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
    
    def planar_calculator(vectors, scalars, ptype, sim_box, indicies, values):
        particle = indicies[0]
        interaction_type = indicies[1] 
        point = values[interaction_type][0:D]      # Point in wall
        normal_vector = values[interaction_type][D:2*D] # Normal vector defining plane of wall
        
        # Calculating full D-dim displacement vector to avoid worrying about new sim_box types in future
        dr = cuda.local.array(shape=D,dtype=numba.float32)
        dist_sq = dist_sq_dr_function(point, vectors[r_id][particle], sim_box, dr)
        dist = numba.float32(0.0)
        for k in range(D):
            dist += dr[k]*normal_vector[k]
        if dist<values[interaction_type][-1]: # Last index is the cut-off
            u, s, umm = potential_function(abs(dist), values[interaction_type][2*D:]) # abs: potential symmetric around wall
        
            for k in range(D):
                cuda.atomic.add(vectors, (f_id, particle, k), -normal_vector[k]*dist*s) # Force
            cuda.atomic.add(scalars, (particle, w_id), dist**2*s)                               # Virial
            cuda.atomic.add(scalars, (particle, u_id), u)                                       # Potential enerrgy 
            lap = numba.float32(1-D)*s + umm                                                    # Laplacian  
            cuda.atomic.add(scalars, (particle, lap_id), lap)               
        
        return
    return planar_calculator

def setup_planar_interactions(configuration, potential_function, potential_params_list, particles_list, point_list, normal_vector_list, compute_plan, verbose=True):
    D = configuration.D
    num_types = len(potential_params_list)
    assert len(particles_list) == num_types
    assert len(point_list) == num_types
    assert len(normal_vector_list) == num_types

    total_number_indicies = 0 
    for particles in particles_list:
        total_number_indicies += particles.shape[0] 

    indicies = np.zeros((total_number_indicies, 2), dtype=np.int32)    
    params = np.zeros((num_types, 2*D+len(potential_params_list[0])), dtype=np.float32)
    
    start_index = 0  
    for interaction_type in range(num_types):
        next_start_index = start_index + len(particles_list[interaction_type])
        indicies[start_index:next_start_index, 0] = particles_list[interaction_type] 
        indicies[start_index:next_start_index, 1] = interaction_type 
        start_index = next_start_index
        
        params[interaction_type,0:D] = point_list[interaction_type]
        params[interaction_type,D:2*D] = normal_vector_list[interaction_type] # Normalize it!
        params[interaction_type,2*D:] = potential_params_list[interaction_type] 
    
    calculator = make_planar_calculator(configuration, potential_function)
    interactions = make_fixed_interactions(configuration, calculator, compute_plan, verbose=verbose)
    d_indicies = cuda.to_device(indicies)
    d_params = cuda.to_device(params)
    interaction_params = (d_indicies, d_params)

    return interactions, interaction_params
     
