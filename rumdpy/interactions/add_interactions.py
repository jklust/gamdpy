import numpy as np
import numba
import math
from numba import cuda

def add_interactions_list(configuration, interactions_list, compute_plan, compute_flags, verbose=True,):
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


def add_interactions_list_experimental(configuration, interactions_list, compute_plan, compute_flags, verbose=True,):
    gridsync = compute_plan['gridsync']
    num_interactions = len(interactions_list)
    assert 0 < num_interactions <= 5
    
    interaction_params_list = []
    interaction_kernel_list = []
    
    for interaction in interactions_list:
        interaction_params_list.append(interaction.get_params(configuration, compute_plan, verbose=verbose))
        interaction_kernel_list.append(interaction.get_kernel(configuration, compute_plan, compute_flags, verbose=verbose))

    params = tuple(interaction_params_list)
    kernels = tuple(interaction_kernel_list)

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            #for i in range(num_interactions):                                                     # Works only with single interaction
            #    kernels[i](grid, vectors, scalars, ptype, sim_box, interaction_parameters[i])
            #    grid.sync()
            kernels[0](grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])          # Gives NumbaExperimentalFeatureWarning
            if num_interactions>1:
                grid.sync() # Not always necessary !!!
                kernels[1](grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                grid.sync() # Not always necessary !!!
                kernels[2](grid, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                grid.sync() # Not always necessary !!!
                kernels[3](grid, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                grid.sync() # Not always necessary !!!
                kernels[4](grid, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return  
        return interactions, params

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            for i in range(len(kernels)):
                kernels[i](0, vectors, scalars, ptype, sim_box, interaction_parameters[i])
            return
        return interactions, params

