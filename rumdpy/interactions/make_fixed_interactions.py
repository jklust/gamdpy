import numba
from numba import cuda


def make_fixed_interactions(configuration, fixed_potential, compute_plan, verbose=True, ):
    """ Generate a kernel for fixed interactions between particles. """
    # Unpack parameters from configuration and compute_plan
    D, num_part = configuration.D, configuration.N
    pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating fixed interactions for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')

    # Prepare user-specified functions for inclusion in kernel(s)
    # NOTE: Include check they can be called with right parameters and returns the right number and type of parameters 

    potential_calculator = numba.njit(fixed_potential)

    def fixed_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
        indices, values = interaction_parameters
        num_interactions = indices.shape[0]
        num_threads = num_blocks * pb * tp

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        my_t = cuda.threadIdx.y
        #global_id = (my_block*pb + local_id)*tp + my_t
        global_id = (my_block * pb + local_id) + my_t * cuda.blockDim.x * cuda.gridDim.x  # Faster

        for index in range(global_id, num_interactions, num_threads):
            potential_calculator(vectors, scalars, ptype, sim_box, indices[index], values)
        return

    fixed_interactions = cuda.jit(device=gridsync)(fixed_interactions)

    if gridsync:
        return fixed_interactions  # return device function
    else:
        return fixed_interactions[num_blocks, (pb, tp)]  # return kernel, incl. launch parameters
