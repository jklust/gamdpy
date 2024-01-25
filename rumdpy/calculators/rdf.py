import numpy as np
import numba
import math
from numba import cuda

#############################################################
#### Radial Distribution Function  
#############################################################

def make_rdf_calculator(configuration, pair_potential, compute_plan, full_range, verbose=True):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    UtilizeNIII = compute_plan['UtilizeNIII']
    gridsync = compute_plan['gridsync']
    num_blocks = (num_part - 1) // pb + 1

    # Unpack indicies for vectors and scalars    
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
   
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
    
    # Prepare user-specified functions for inclusion in kernel(s)
    ptype_function = numba.njit(configuration.ptype_function)
    params_function = numba.njit(pair_potential.params_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)


    #@cuda.jit(device=gridsync)
    def rdf_calculator_full(vectors, sim_box, ptype, interaction_parameters, d_gr_bins):
        """ Calculate g(r) fresh
            Kernel configuration: [num_blocks, (pb, tp)]
        """
        params, max_cut, skin, nblist, nbflag, exclusions = interaction_parameters

        num_bins = d_gr_bins.shape[0]  # reading number of bins from size of the device array
        min_box_dim = min(sim_box[0], sim_box[1], sim_box[2])   # max distance for rdf can 0.5*Smallest dimension
        bin_width = (min_box_dim/2)/num_bins

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        if global_id < num_part:
            max_nbs = nblist.shape[1] - 1
            for i in range(0, num_part, pb * tp):
                for j in range(pb):
                    other_global_id = j + i + my_t * pb
                    if other_global_id != global_id and other_global_id < num_part:
                        dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id], sim_box)

                        # Calculate g(r)
                        if dist_sq < (min_box_dim / 2) ** 2:
                            dist = math.sqrt(dist_sq)
                            if dist < min_box_dim / 2:
                                bin_index = int(dist / bin_width)
                                cuda.atomic.add(d_gr_bins, bin_index, 1)

        return

    #@cuda.jit(device=gridsync)
    def rdf_calculator_from_nblist(vectors, sim_box, ptype, interaction_parameters, d_gr_bins):
        """ Calculate g(r) using neighbor-list
            Kernel configuration: [num_blocks, (pb, tp)]
        """
        params, max_cut, skin, nblist, nbflag, exclusions = interaction_parameters

        num_bins = d_gr_bins.shape[0]  # reading number of bins from size of the device array
        min_box_dim = min(sim_box[0], sim_box[1], sim_box[2])   
       
        #bin_width = (min_box_dim/2)/num_bins
        bin_width = max_cut/num_bins

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x 
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        max_nbs = nblist.shape[1] - 1
        
        if global_id < num_part:
            my_type = ptype_function(global_id, ptype)
        
        cuda.syncthreads() 

        if global_id < num_part:
            num_neighbors = nblist[global_id, max_nbs]

            for i in range(my_t, num_neighbors, tp):
                other_id = nblist[global_id, i]  # Neighbor particle index
                other_type = ptype_function(other_id, ptype)
              
                # Calculate the squared distance between particles
                dist_sq = dist_sq_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box)
                              
                # Calculate g(r)
                ij_params = params_function(my_type, other_type, params)
                cut = ij_params[-1]
                if dist_sq < cut*cut:
                    dist = math.sqrt(dist_sq)
                    if dist < min_box_dim / 2:
                        bin_index = int(dist / bin_width)
                        if UtilizeNIII:
                            cuda.atomic.add(d_gr_bins, bin_index, 2)       #particles a and b are listed as neighbours only once in the nblist 
                        else:
                            cuda.atomic.add(d_gr_bins, bin_index, 1)

        return
    
    if full_range:
        return cuda.jit(device=0)(rdf_calculator_full)[num_blocks, (pb, tp)]
    else:
        return cuda.jit(device=0)(rdf_calculator_from_nblist)[num_blocks, (pb, tp)]

def normalize_and_save_gr(gr_bins, c1, interaction_params, full_range, steps, filename):

    max_cut = interaction_params[1]
    num_bins = gr_bins.shape[0]
    min_box_dim = min(c1.simbox.lengths[0], c1.simbox.lengths[1], c1.simbox.lengths[2])
    num_gr_updates = steps

    if full_range:
        bin_width = (min_box_dim / 2) / num_bins
    else:
        bin_width = max_cut / num_bins

    # Normalize the g(r) lengths
    rho = c1.N / (c1.simbox.lengths[0] * c1.simbox.lengths[1] * c1.simbox.lengths[2])
    for i in range(len(gr_bins)):
        r_outer = (i + 1) * bin_width
        r_inner = i * bin_width
        shell_volume = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        expected_num = rho * shell_volume
        gr_bins[i] /= (expected_num * num_gr_updates * c1.N)

  
    # Save lengths to file
    distances = np.arange(0, len(gr_bins)) * bin_width
    data_to_save = np.column_stack((distances, gr_bins))
    np.savetxt(filename, data_to_save, comments='', fmt='%f')
 
    return data_to_save


def normalize_and_save_gr_multi(gr_bins, c1, interaction_params, full_range, filename):

    max_cut = interaction_params[1]
    num_bins = gr_bins.shape[1]
    min_box_dim = min(c1.simbox.lengths[0], c1.simbox.lengths[1], c1.simbox.lengths[2])
    print(min_box_dim)

    if full_range:
        bin_width = (min_box_dim / 2) / num_bins
    else:
        bin_width = max_cut / num_bins

    # Normalize the g(r) lengths
    rho = c1.N / np.prod(c1.simbox.lengths)
    print(rho, bin_width)
    for i in range(gr_bins.shape[1]): # Normalize one bin/distance at a time
        r_outer = (i + 1) * bin_width
        r_inner = i * bin_width
        shell_volume = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        expected_num = rho * shell_volume
        gr_bins[:,i] /= (expected_num * c1.N)
  
    # Save lengths to file
    distances = np.arange(0, gr_bins.shape[1]) * bin_width
    data_to_save = np.column_stack((distances, gr_bins.T))
    np.savetxt(filename, data_to_save, comments='', fmt='%f')
 
    return data_to_save
