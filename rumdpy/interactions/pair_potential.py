import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import rumdpy as rp

class PairPotential():
    """ Pair potential """

    def __init__(self, pairpotential_function, params, max_num_nbs, exclusions=None):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]            # default: read from params array
            return result            
    
        self.pairpotential_function = pairpotential_function
        self.params_function = params_function
        self.params_user = params
        self.exclusions = exclusions 
        self.max_num_nbs = max_num_nbs

    def convert_user_params(self):
        # Upgrade any scalar parameters to 1x1 numpy array
        num_params = len(self.params_user)
        params_list = []
        for parameter in self.params_user:
            if np.isscalar(parameter):
                params_list.append(np.ones((1,1))*parameter)
            else:
                params_list.append(np.array(parameter, dtype=np.float32))

        # Ensure all parameters are the right format (num_types x num_types) numpy arrays
        num_types = params_list[0].shape[0]
        for parameter in params_list:
            assert len(parameter.shape) == 2
            assert parameter.shape[0] == num_types
            assert parameter.shape[1] == num_types

        # Convert params to the format required by kernels (num_types x num_types) array of tuples (p0, p1, ..., cutoff)
        params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                plist = []
                for parameter in params_list:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(params_list[-1]))

        return params, max_cut
               
    def plot(self, xlim=None, ylim=(-3,6), figsize=(8,4), names=None):
        params, max_cut = self.convert_user_params()
        num_types = len(params[0])
        if names==None:
            names = np.arange(num_types)
        plt.figure(figsize=figsize)
        for i in range(num_types):
            for j in range(num_types):
                r = np.linspace(0, params[i,j][-1], 1000)
                u, s, lap = self.pairpotential_function(r, params[i,j])
                plt.plot(r, u, label=f'{names[i]} - {names[j]}')
        if xlim!=None:
            plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Pair distance')
        plt.ylabel('Pair potential')
        plt.legend()
        plt.show()
        
  
    def get_params(self, configuration, compute_plan, verbose=False):
        
        self.params, max_cut = self.convert_user_params()
        self.d_params = cuda.to_device(self.params)

        self.nblist = rp.NbList2(configuration, self.exclusions, self.max_num_nbs)        
        nblist_params = self.nblist.get_params(max_cut, compute_plan, verbose)

        return (self.d_params, self.nblist.d_nblist, nblist_params)

    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        num_cscalars = 3 # TODO: deal with this

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1  

        if verbose:
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
            if compute_stresses:
                print('\tIncluding computation of stress tensor in pair potential')
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        if compute_stresses:
            sx_id, sy_id, sz_id = [configuration.vectors.indices[key] for key in ['sx', 'sy', 'sz']] # D=3 !!!
        u_id, w_id, lap_id, m_id = [configuration.sid[key] for key in ['u', 'w', 'lap', 'm']]


        pairpotential_function = self.pairpotential_function
    
        if UtilizeNIII:
            virial_factor_NIII = numba.float32( 1.0/configuration.D)
            def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
                u, s, umm = pairpotential_function(ij_dist, ij_params)
                for k in range(D):
                    cuda.atomic.add(f, (other_id, k), dr[k]*s)
                    my_f[k] = my_f[k] - dr[k]*s                         # Force
                    cscalars[w_id] += dr[k]*dr[k]*s*virial_factor_NIII  # Virial
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] -= dr[k]*dr[k2]*s

                cscalars[u_id] += u                                      # Potential energy
                cscalars[lap_id] += (numba.float32(1-D)*s + umm)*numba.float32( 2.0 ) # Laplacian 


                return
            
        else:
            virial_factor = numba.float32( 0.5/configuration.D )
            def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
                u, s, umm = pairpotential_function(ij_dist, ij_params)
                half = numba.float32(0.5)
                for k in range(D):
                    my_f[k] = my_f[k] - dr[k]*s                         # Force
                    cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] -= half*dr[k]*dr[k2]*s      # stress tensor
                cscalars[u_id] += half*u                                # Potential energy
                cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
                return

        ptype_function = numba.njit(configuration.ptype_function)
        params_function = numba.njit(self.params_function)
        pairpotential_calculator = numba.njit(pairpotential_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
        dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)
    
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
            if compute_stresses:
                my_stress = cuda.local.array(shape=(D,D), dtype=numba.float32)
            else:
                my_stress = cuda.local.array(shape=(1,1), dtype=numba.float32)
        
            if global_id < num_part:
                for k in range(D):
                    #my_r[k] = r[global_id, k]
                    my_f[k] = numba.float32(0.0)
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] = numba.float32(0.0)
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
                        pairpotential_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, my_stress, vectors[f_id], other_id)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
                    if compute_stresses:
                        #for k2 in range(D):
                        cuda.atomic.add(vectors[sx_id], (global_id, k), my_stress[0,k])
                        cuda.atomic.add(vectors[sy_id], (global_id, k), my_stress[1,k])
                        cuda.atomic.add(vectors[sz_id], (global_id, k), my_stress[2,k])
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            return 
        
        nblist_check_and_update = self.nblist.get_kernel(configuration, compute_plan, verbose)

        if gridsync:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                grid.sync()
                calc_forces(vectors, scalars, ptype, sim_box, nblist, params)
                return
            return compute_interactions
        
        else:
            # A python function, making several kernel calls to syncronize  
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                calc_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params)
                return
            return compute_interactions


