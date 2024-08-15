import numpy as np
import numba
from numba import cuda

class NbList2():
    
    def __init__(self, configuration, exclusions, max_num_nbs):
        self.nblist = np.zeros((configuration.N, max_num_nbs+1), dtype=np.int32) 
        self.nbflag = np.zeros(3, dtype=np.int32)
        self.r_ref = np.zeros_like(configuration['r']) # Inherits also data type
        self.exclusions = exclusions  # Should be able to be a list (eg from bonds, angles, etc), and merge        

    def copy_to_device(self):
        self.d_nblist = cuda.to_device(self.nblist)
        self.d_nbflag = cuda.to_device(self.nbflag)
        self.d_r_ref = cuda.to_device(self.r_ref)
        self.d_exclusions = cuda.to_device(self._exclusions)

    def get_params(self, max_cut, compute_plan, verbose=True):
        self.max_cut = max_cut
        self.skin, = [compute_plan[key] for key in ['skin']]

        #print('NbList.exclusions:\n', self.exclusions)
        if type(self.exclusions) == np.ndarray: # Don't change user-set properties
            self._exclusions = self.exclusions.copy()
        else:
            self._exclusions = np.zeros((self.r_ref.shape[0], 2), dtype=np.int32)
        self.copy_to_device()                     
        return (np.float32(self.max_cut), np.float32(self.skin), self.d_nbflag, self.d_r_ref, self.d_exclusions)

    def get_kernel(self, configuration, compute_plan, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1
        compute_stresses = configuration.compute_stresses

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        if compute_stresses:
            sx_id, sy_id, sz_id = [configuration.vectors.indices[key] for key in ['sx', 'sy', 'sz']]

        # JIT compile functions to be compiled into kernel
        dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)
        dist_moved_sq_function = numba.njit(configuration.simbox.dist_moved_sq_function)

        @cuda.jit( device=gridsync )
        def update_strain_change(sim_box, cut):
            # this could also be something every thread (or at least my_t==0) does for themselves, so avoiding
            # the need for extra synchronization
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y

            if global_id == 0 and my_t == 0:
                strain_change = sim_box[D] - sim_box[D+2] # change in box-shift
                # take PBC into account:
                if strain_change > 0.5*sim_box[0]:
                    strain_change -= sim_box[0]
                if strain_change < -0.5*sim_box[0]:
                    strain_change += sim_box[0]
                strain_change /= sim_box[1] # convert to (xy) strain

                sim_box[D+4] = strain_change
                sim_box[D+5] = abs(strain_change)*cut

        @cuda.jit( device=gridsync )
        def nblist_check(vectors, sim_box, skin, r_ref, nbflag):
            """ Check validity of nblist, i.e. did any particle mode more than skin/2 since last nblist update?
                Each thread-block checks the assigned particles (global_id)
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
                    dist_sq = dist_moved_sq_function(vectors[r_id][global_id], r_ref[global_id], sim_box)
                    if len(sim_box) == D+6:  # LE boundaries (J. Chattoraj. Ph.D. thesis (2011))
                        skin -= sim_box[D+5]
                        if skin < 0:
                            skin = 0. # otherwise wrong when you square!
                    if dist_sq > skin*skin*numba.float32(0.25):
                        nbflag[0] = num_blocks

            if global_id < num_part and my_t==0: # Initializion of forces moved here to make NewtonIII possible 
                for k in range(D):
                    vectors[f_id][global_id, k] = numba.float32(0.0)
                    if  compute_stresses:
                        vectors[sx_id][global_id, k] =  numba.float32(0.0)
                        vectors[sy_id][global_id, k] =  numba.float32(0.0)
                        vectors[sz_id][global_id, k] =  numba.float32(0.0)
            return
   
        @cuda.jit(device=gridsync)
        def nblist_update(vectors, sim_box, cut_plus_skin, nbflag, nblist, r_ref, exclusions):
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
                    
                cuda.syncthreads() # wait for nblist[global_id, max_nbs] to be ready ready
                
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
                        r_ref[global_id, k] = vectors[r_id][global_id, k]   # Store positions for wich nblist was updated ( used in nblist_check() ) 
                if local_id == 0 and my_t==0:
                    cuda.atomic.add(nbflag, 0, -1)              # nbflag[0] = 0 by when all blocks are done
                if global_id == 0 and my_t==0:
                    cuda.atomic.add(nbflag, 2, 1)               # Count how many updates are done in nbflag[2]
                    # the following is for Lees-Edwards boundary conditions
                    if len(sim_box) == D+6: # we need a more elegant way to check which kind of simbox we have!
                        # (like a generic function for the simbox to update itself after NB rebuild)
                        sim_box[D+2] = sim_box[D]
                        sim_box[D+3] = sim_box[D+1]

                if my_num_nbs >= max_nbs:                       # Overflow detected, nbflag[1] should be checked later, and then
                    cuda.atomic.max(nbflag, 1, my_num_nbs)      # re-allocate larger nb-list, and redo computations from last safe state
    
            return 
        
        if gridsync==True:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters):
                max_cut, skin, nbflag, r_ref, exclusions = nblist_parameters
                if len(sim_box) == D+6: # Lees-Edwards BC
                    update_strain_change(sim_box, max_cut)
                    grid.sync()
                nblist_check(vectors, sim_box, skin, r_ref, nbflag)
                grid.sync()
                nblist_update(vectors, sim_box, max_cut+skin, nbflag, nblist, r_ref, exclusions)
                return
            return check_and_update
        
        else:
            # A python function, making several kernel calls to syncronize  
            def check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters):
                max_cut, skin, nbflag, r_ref, exclusions = nblist_parameters
                if len(sim_box) == D+6: # Lees-Edwards BC
                    update_strain_change[num_blocks, (1, 1)](sim_box, max_cut)
                nblist_check[num_blocks, (pb, 1)](vectors, sim_box, skin, r_ref, nbflag)
                nblist_update[num_blocks, (pb, tp)](vectors, sim_box, max_cut+skin, nbflag, nblist, r_ref, exclusions)
                return
            return check_and_update
