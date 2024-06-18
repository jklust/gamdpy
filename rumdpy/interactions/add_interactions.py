import numpy as np
import numba
import math
from numba import cuda

def add_interactions_bak(configuration, interactions0,  interactions1, compute_plan, verbose=True,):
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
    
def add_interactions_list_bak(configuration, interactions_list, compute_plan, verbose=True,):
    gridsync = compute_plan['gridsync']
    length = len(interactions_list)
    assert length <= 5
    
    i0 = interactions_list[0]
    if len(interactions_list)>1:
        i1 = interactions_list[1]
    if len(interactions_list)>2:
        i2 = interactions_list[2]
    if len(interactions_list)>3:
        i3 = interactions_list[3]
    if len(interactions_list)>4:
        i4 = interactions_list[4]

    interaction_params_list = []
    for interaction in interactions_list:
        interaction_params_list.append(interaction['interaction_params'])
    
    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if length>1:
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
        return interactions, tuple(interaction_params_list)

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if length>1:
                i1(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if length>2:
                i2(0, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if length>3:
                i3(0, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if length>4:
                i4(0, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return interactions, tuple(interaction_params_list)

def add_interactions_list(configuration, interactions_list, compute_plan, compute_stresses=False, verbose=True,):
    gridsync = compute_plan['gridsync']
    num_interactions = len(interactions_list)
    assert 0 < num_interactions <= 5
    
    interaction_params_list = []
    for interaction in interactions_list:
        interaction_params_list.append(interaction.get_params(configuration, compute_plan, verbose=verbose))

    i0 = interactions_list[0].get_kernel(configuration, compute_plan, compute_stresses, verbose=verbose)
    if num_interactions>1:
        i1 = interactions_list[1].get_kernel(configuration, compute_plan, compute_stresses, verbose=verbose)
    if num_interactions>2:
        i2 = interactions_list[2].get_kernel(configuration, compute_plan, compute_stresses, verbose=verbose)
    if num_interactions>3:
        i3 = interactions_list[3].get_kernel(configuration, compute_plan, compute_stresses, verbose=verbose)
    if num_interactions>4:
        i4 = interactions_list[4].get_kernel(configuration, compute_plan, compute_stresses, verbose=verbose)

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if num_interactions>1:
                grid.sync() # Not always necesarry !!!
                i1(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                grid.sync() # Not always necesarry !!!
                i2(grid, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                grid.sync() # Not always necesarry !!!
                i3(grid, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                grid.sync() # Not always necesarry !!!
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

