import numpy as np
import numba
from numba import cuda
import math

def make_integrator(configuration, integration_step, compute_interactions, compute_plan, verbose=True,):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part-1)//pb + 1  
 
    if gridsync:
        # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
        @cuda.jit       
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, steps):
            grid = cuda.cg.this_grid()
            for i in range(steps):
                compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                grid.sync()
                integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params)
                grid.sync()
            return
        return integrator[num_blocks, (pb, tp)]
         
    else:
        
        # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, steps):
            for i in range(steps):
                compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                integration_step(0, vectors, scalars, r_im, sim_box, integrator_params)
            return
        return integrator
    
    return 


def make_step_nve(configuration, compute_plan, verbose=True,):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']   
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part-1)//pb + 1  
    
    if verbose:
        print(f'Generating NVE integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')      

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
        
    #@cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync) 
    #@cuda.jit(device=gridsync) 
    def step_nve(grid, vectors, scalars, r_im, sim_box, integrator_params):
        """ Make one NVE timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]        
        """
    
        dt, = integrator_params
    
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y
        
        if global_id < num_part and my_t==0:
            my_r = vectors[r_id][global_id]
            my_v = vectors[v_id][global_id]
            my_f = vectors[f_id][global_id]
            my_m = scalars[global_id][m_id]
            my_k = numba.float32(0.0) # Kinetic energy
            my_fsq = numba.float32(0.0) # force squared energy
            
            for k in range(D):
                my_fsq += my_f[k]*my_f[k]
                my_v[k] += numba.float32(0.5)*my_f[k]/my_m*dt
                my_k += numba.float32(0.5)*my_m*my_v[k]*my_v[k]
                my_v[k] += numba.float32(0.5)*my_f[k]/my_m*dt
                my_r[k] += my_v[k]*dt 
                if my_r[k]*numba.float32(2.0) > sim_box[k]:
                    my_r[k] -= sim_box[k]
                    r_im[global_id, k] += 1
                if my_r[k]*numba.float32(2.0) < -sim_box[k]:
                    my_r[k] += sim_box[k]
                    r_im[global_id, k] -= 1
                #if my_r[k]*numba.float32(2.0) > sim_box[k] or  my_r[k]*numba.float32(2.0) < -sim_box[k]:
                #    print(global_id, k, my_r[0], my_r[1], my_r[2])
                #vectors[r_id][global_id,k] = my_r[k]
            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return
    
    if gridsync:
        return cuda.jit(device=gridsync)(step_nve)                        # return device function
    else:
        return cuda.jit(device=gridsync)(step_nve)[num_blocks, (pb, 1)]   # return kernel, incl. launch parameters

    
    
def make_step_nvt(configuration, compute_plan, verbose=True,):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']   
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part-1)//pb + 1  
    
    if verbose:
        print(f'Generating NVT integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks*pb}')
        print(f'\tNumber of threads {num_blocks*pb*tp}')

    # Unpack indicies for vectors and scalars    
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
        
    #@cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync) 
    #@cuda.jit(device=gridsync) 
    def step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params):
        """ Make one NVT timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]        
        """
        
        dt, target_temperature, omega2, degrees, thermostat_state = integrator_params # Put more in thermostat_state?

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y
        
        factor = np.float32(0.5) * thermostat_state[0] * dt
        plus  = np.float32(1.) / ( np.float32(1.) + factor ) # Possibly change to exp(...)
        minus = np.float32(1.) - factor # Possibly change to exp(...)
        
        if global_id < num_part and my_t==0:
            my_r = vectors[r_id][global_id]
            my_v = vectors[v_id][global_id]
            my_f = vectors[f_id][global_id]
            my_m = scalars[global_id][m_id]
            my_k = numba.float32(0.0) # Kinetic energy
            my_fsq = numba.float32(0.0) # force squared
            
            for k in range(D):
                my_fsq += my_f[k]*my_f[k]
                #my_v[k] += numba.float32(0.5)*my_f[k]/my_m*dt
                my_v[k] = plus * ( minus*my_v[k] + my_f[k]/my_m*dt )
                my_k += numba.float32(0.5)*my_m*my_v[k]*my_v[k]
                #my_v[k] += numba.float32(0.5)*my_f[k]/my_m*dt
                my_r[k] += my_v[k]*dt 
                if my_r[k]*numba.float32(2.0) > sim_box[k]:
                    my_r[k] -= sim_box[k]
                    r_im[global_id, k] += 1
                if my_r[k]*numba.float32(2.0) < -sim_box[k]:
                    my_r[k] += sim_box[k]
                    r_im[global_id, k] -= 1
                #vectors[r_id][global_id,k] = my_r[k]
            cuda.atomic.add(thermostat_state, 1, my_k) # Probably slow! Spread out over num_blocks terms?
            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return
    
    def update_thermostat_state(integrator_params):
        dt, target_temperature, omega2, degrees, thermostat_state = integrator_params # Put more in thermostat_state?
        # Some of these can be compiled in, but will be less flexible
        
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y

        if global_id==0 and my_t==0:
            ke_deviation = np.float32(2.0) * thermostat_state[1] / (degrees*target_temperature) - np.float32(1.0)
            thermostat_state[0] += dt * omega2 * ke_deviation
            thermostat_state[1] = np.float32(0.)
        return 
        
    step_nvt = cuda.jit(device=gridsync)(step_nvt)                        
    update_thermostat_state = cuda.jit(device=gridsync)(update_thermostat_state)    
    
    if gridsync:

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params):
            step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params)
            grid.sync()
            update_thermostat_state(integrator_params)
            return
        return cuda.jit(device=gridsync)(step)
    
    else:
        
        def step(grid, vectors, scalars, r_im, sim_box, integrator_params):
            step_nvt[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params)
            update_thermostat_state[1, (1, 1)](integrator_params)
            return
        return step


def make_run_langevin_nvt(configuration, compute_plan, verbose=True, ):
    raise NotImplementedError('make_run_langevin_nvt is not implemented yet')

    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating NVT langevin integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')

    # Unpack indices for vectors and scalars
    for key in configuration.vid:
        exec(f'{key}_id = {configuration.vid[key]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    # @cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync)
    # @cuda.jit(device=gridsync)
    def step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params):
        """ Make one timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]
        """

        dt, = integrator_params

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        if global_id < num_part and my_t == 0:
            my_r = vectors[r_id][global_id]
            my_v = vectors[v_id][global_id]
            my_f = vectors[f_id][global_id]
            my_m = scalars[global_id][m_id]
            my_k = numba.float32(0.0)  # Kinetic energy
            my_fsq = numba.float32(0.0)  # force squared energy

            for k in range(D):
                my_fsq += my_f[k] * my_f[k]
                my_v[k] += numba.float32(0.5) * my_f[k] / my_m * dt
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]
                my_v[k] += numba.float32(0.5) * my_f[k] / my_m * dt
                my_r[k] += my_v[k] * dt
                if my_r[k] * numba.float32(2.0) > sim_box[k]:
                    my_r[k] -= sim_box[k]
                    r_im[global_id, k] += 1
                if my_r[k] * numba.float32(2.0) < -sim_box[k]:
                    my_r[k] += sim_box[k]
                    r_im[global_id, k] -= 1
                # if my_r[k]*numba.float32(2.0) > sim_box[k] or  my_r[k]*numba.float32(2.0) < -sim_box[k]:
                #    print(global_id, k, my_r[0], my_r[1], my_r[2])
                # vectors[r_id][global_id,k] = my_r[k]
            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return

    if gridsync:
        return cuda.jit(device=gridsync)(step_nvt)  # return device function
    else:
        return cuda.jit(device=gridsync)(step_nvt)[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters


def setup_integrator_nvt(configuration, interactions, tau, dt, T, compute_plan, verbose=True):
    integrator_step = make_step_nvt(configuration, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan, verbose=verbose)

    dt = np.float32(dt)
    T = np.float32(T)
    omega2 = np.float32(4.0*np.pi*np.pi/tau/tau)
    degrees = configuration.N*configuration.D - configuration.D
    thermostat_state = np.zeros(2, dtype=np.float32)
    d_thermostat_state = cuda.to_device(thermostat_state)
    integrator_params =  (dt, T, omega2, degrees,  d_thermostat_state)
    
    return integrate, integrator_params

    