import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt


class PairPotential():
    """ Pair potential """

    def __init__(self, configuration, pairpotential_function, params, exclusions, max_num_nbs, compute_plan):
        N = configuration.N
        D = configuration.D
        self.exclusions = exclusions 
        
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
        self.params_user = np.array(params,dtype=np.float32)
        self.nblist = NbList(N, max_num_nbs)
        return
    
    def copy_to_device(self):
        self.d_params = cuda.to_device(self.params)
        self.nblist.copy_to_device()
        
    def plot(self, ylim=(-3,6), figsize=(8,4), names=None):
        num_types = self.params_user.shape[0]
        if names==None:
            names = np.arange(num_types)
        plt.figure(figsize=figsize)
        for i in range(num_types):
            for j in range(num_types):
                r = np.linspace(0, self.params_user[i,j][-1], 1000)
                u, s, lap = self.pairpotential_function(r, self.params_user[i,j])
                plt.plot(r, u, label=f'{names[i]} - {names[j]}')
        plt.ylim(ylim)
        plt.xlabel('Pair distance')
        plt.ylabel('Pair potential')
        plt.legend()
        plt.show()
        
    def get_interactions(self, configuration, exclusions, compute_plan, verbose=True):
   
        if exclusions == None:
            exclusions = cuda.to_device(np.zeros((configuration.N, 2), dtype=np.int32))

        if verbose:
            print(f'Setting up pair interactions for {configuration.N} particles in {configuration.D} dimensions:')
            print(f'\tcompute_plan: {compute_plan}')   
            print(f'\tparameters:\n{self.params_user}')   

        # Setup params array in right format
        num_types = self.params_user.shape[0]
        assert num_types == self.params_user.shape[1]
        num_params = len(self.params_user[0,0])
        
        # Make params in the right format
        self.params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                self.params[i,j] = tuple(self.params_user[i,j])
        self.copy_to_device()
                
        max_cut = np.float32(np.max(self.params_user[:,:,-1]))
        num_cscalars = 3
        interactions = make_interactions(configuration, self, num_cscalars=num_cscalars,
                                         compute_plan=compute_plan, verbose=verbose,)
        
        # Should be able to take a list of exclusions (eg from bonds, angles, etc), and merge 
        interaction_params = (self.d_params, max_cut, np.float32(compute_plan['skin']), 
                              self.nblist.d_nblist,  self.nblist.d_nbflag, exclusions)

        return {'interactions': interactions, 
                'interaction_params': interaction_params}

    def get_params(self, configuration, compute_plan, verbose=True):
        exclusions = self.exclusions # Don't change user-set propertys
        if exclusions == None:
            d_exclusions = cuda.to_device(np.zeros((configuration.N, 2), dtype=np.int32))
        else:
            d_exclusions = cuda.to_device(exclusions)

        # Setup params array in right format
        num_types = self.params_user.shape[0]
        assert num_types == self.params_user.shape[1]
        num_params = len(self.params_user[0,0])
        
        # Make params in the right format
        self.params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                self.params[i,j] = tuple(self.params_user[i,j])
        self.copy_to_device()
                
        max_cut = np.float32(np.max(self.params_user[:,:,-1]))
        
        # Should be able to take a list of exclusions (eg from bonds, angles, etc), and merge 
        return (self.d_params, max_cut, np.float32(compute_plan['skin']), 
                              self.nblist.d_nblist,  self.nblist.d_nbflag, d_exclusions)

    def get_kernel(self, configuration, compute_plan, verbose=True):
        num_cscalars = 3
        return make_interactions(configuration, self, num_cscalars=num_cscalars,
                                         compute_plan=compute_plan, verbose=verbose,)

####################################################
### NBlist
####################################################'

class NbList():
    def __init__(self, num_part, max_num_nbs):
        self.nblist = np.zeros((num_part, max_num_nbs+1), dtype=np.int32) # Might skip this to save memory on host
        self.nbflag = np.zeros(3, dtype=np.int32)

    def copy_to_device(self):
        self.d_nblist = cuda.to_device(self.nblist)
        self.d_nbflag = cuda.to_device(self.nbflag)

def make_interactions(configuration, pair_potential, num_cscalars, compute_plan, verbose=True,):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    UtilizeNIII = compute_plan['UtilizeNIII']
    num_blocks = (num_part-1)//pb + 1    

    if verbose:
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')      

    # Unpack indicies for vectors and scalars    
    for col in configuration.vectors.column_names:
            exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    # Prepare user-specified functions for inclusion in kernel(s)
    # NOTE: Include check they can be called with right parameters and returns the right number and type of parameters 

    pairpotential_function = pair_potential.pairpotential_function
    
    if compute_plan['UtilizeNIII']:
        virial_factor_NIII = numba.float32( 1.0/configuration.D)
        def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, f, other_id):
            u, s, umm = pairpotential_function(ij_dist, ij_params)
            for k in range(D):
                cuda.atomic.add(f, (other_id, k), dr[k]*s)
                my_f[k] = my_f[k] - dr[k]*s                         # Force
                cscalars[w_id] += dr[k]*dr[k]*s*virial_factor_NIII  # Virial
            cscalars[u_id] += u                                      # Potential energy
            cscalars[lap_id] += (numba.float32(1-D)*s + umm)*numba.float32( 2.0 ) # Laplacian 
            return
            
    else:
            
        virial_factor = numba.float32( 0.5/configuration.D )
        def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, f, other_id):
            u, s, umm = pairpotential_function(ij_dist, ij_params)
            for k in range(D):
                my_f[k] = my_f[k] - dr[k]*s                         # Force
                cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
            cscalars[u_id] += u*numba.float32( 0.5 )                # Potential energy
            cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
            return

    ptype_function = numba.njit(configuration.ptype_function)
    params_function = numba.njit(pair_potential.params_function)
    pairpotential_calculator = numba.njit(pairpotential_calculator)
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


