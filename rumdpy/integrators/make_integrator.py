import numpy as np
import numba
from numba import cuda
import math

def make_integrator(configuration, integration_step, compute_interactions, compute_plan, verbose=True, ):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if gridsync:
        # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
        @cuda.jit
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, time_zero, steps):
            grid = cuda.cg.this_grid()
            time = time_zero
            for i in range(steps):
                compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                grid.sync()
                integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                time += integrator_params[0]  # dt. Should be more specific
            return

        return integrator[num_blocks, (pb, tp)]

    else:

        # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, time_zero, steps):
            time = time_zero
            for i in range(steps):
                compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                time += integrator_params[0]  # dt. Should be more specific
            return

        return integrator

    return


def make_integrator_with_output(configuration, integration_step, compute_interactions, output_calculator, conf_saver, compute_plan, verbose=True, ):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1
    
    if output_calculator != None:
        output_calculator = cuda.jit(device=gridsync)(numba.njit(output_calculator))
    if conf_saver != None:
        conf_saver = cuda.jit(device=gridsync)(numba.njit(conf_saver))


    if gridsync:
        # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
        @cuda.jit
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, output_array, conf_array, time_zero, steps):
            grid = cuda.cg.this_grid()
            time = time_zero
            for step in range(steps):
                compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                if conf_saver != None:
                    conf_saver(grid, vectors, scalars, r_im, sim_box, conf_array, step)
                grid.sync()
                time = time_zero + step*integrator_params[0]
                integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                if output_calculator != None:
                    output_calculator(grid, vectors, scalars, r_im, sim_box, output_array, step)
                grid.sync()
                #time += integrator_params[0]  # dt. ! Dont do many additions like this
            if conf_saver != None:
                conf_saver(grid, vectors, scalars, r_im, sim_box, conf_array, steps) # Save final configuration (if conditions fullfiled)
            return

        return integrator[num_blocks, (pb, tp)]

    else:

        # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, time_zero, steps):
            time = time_zero
            for i in range(steps):
                compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                time = time_zero + step*integrator_params[0]
                integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                
            return

        return integrator

    return
