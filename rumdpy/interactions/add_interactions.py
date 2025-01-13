import numpy as np
import numba
import math
from numba import cuda
from rumdpy.interactions import Interaction


def add_interactions_list_old(configuration, interactions_list, compute_plan, compute_flags, verbose=True,):
    gridsync = compute_plan['gridsync']
    num_interactions = len(interactions_list)
    assert 0 < num_interactions <= 5
    
    interaction_params_list = []
    for interaction in interactions_list:
        interaction_params_list.append(interaction.get_params(configuration, compute_plan, verbose=verbose))

    i0 = interactions_list[0].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>1:
        i1 = interactions_list[1].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>2:
        i2 = interactions_list[2].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>3:
        i3 = interactions_list[3].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>4:
        i4 = interactions_list[4].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if num_interactions>1:
                grid.sync() # Not always necessary !!!
                i1(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                grid.sync() # Not always necessary !!!
                i2(grid, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                grid.sync() # Not always necessary !!!
                i3(grid, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                grid.sync() # Not always necessary !!!
                i4(grid, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return interactions, tuple(interaction_params_list)

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if num_interactions>1:
                i1(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                i2(0, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                i3(0, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                i4(0, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return interactions, tuple(interaction_params_list)


def merge_interactions(configuration, kernel, params, interactionB: Interaction, compute_plan, compute_flags, verbose=True,):
    gridsync = compute_plan['gridsync']

    paramsB = interactionB.get_params(configuration, compute_plan, verbose=verbose)
    kernelB = interactionB.get_kernel(configuration, compute_plan, compute_flags, verbose=verbose) 

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            kernel(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])        
            grid.sync() # Not always necessary !!!
            kernelB(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])        
            return
        return interactions, (params, paramsB, )
    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            kernel(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])        
            kernelB(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])        
            return
        return interactions, (params, paramsB, )


def add_interactions_list(configuration, interactions_list, compute_plan, compute_flags, verbose=True,):
    gridsync = compute_plan['gridsync']
    assert 0 < len(interactions_list)

    # Setup first interaction and cuda.jit it if gridsync is used for syncronization
    params = interactions_list[0].get_params(configuration, compute_plan, verbose=verbose)
    kernel = interactions_list[0].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if gridsync:
        kernel = cuda.jit( device=gridsync )(kernel)
    
    # Merge in the rest of the interaction (maximum recursion depth might set a maximum for number of interactions)
    for i in range(1, len(interactions_list)):
        kernel, params = merge_interactions(configuration, kernel, params, interactions_list[i], compute_plan, compute_flags)

    return kernel, params