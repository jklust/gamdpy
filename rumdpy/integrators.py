import numpy as np
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

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
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, time_zero, steps):
            grid = cuda.cg.this_grid()
            time = time_zero
            for i in range(steps):
                compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                grid.sync()
                integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                time += integrator_params[0] # dt. Should be more specific
            return
        return integrator[num_blocks, (pb, tp)]
         
    else:
        
        # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
        def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, time_zero, steps):
            time = time_zero
            for i in range(steps):
                compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                time += integrator_params[0] # dt. Should be more specific
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
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
        
    #@cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync) 
    #@cuda.jit(device=gridsync) 
    def step_nve(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
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

    
    
def make_step_nvt(configuration, temperature_function, compute_plan, verbose=True,):
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
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
        
    temperature_function = numba.njit(temperature_function)
    
    #@cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync) 
    #@cuda.jit(device=gridsync) 
    def step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NVT timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]
        """
        
        dt, omega2, degrees, thermostat_state = integrator_params # Put more in thermostat_state?

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
                if my_r[k]*numba.float32(2.0) > sim_box[k]: # Should be controled by function in simbox
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
    
    def update_thermostat_state(integrator_params, time):
        dt, omega2, degrees, thermostat_state = integrator_params # Put more in thermostat_state?
        # Some of these can be compiled in, but will be less flexible
        
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block*pb + local_id
        my_t = cuda.threadIdx.y

        if global_id==0 and my_t==0:
            target_temperature = temperature_function(time)
            ke_deviation = np.float32(2.0) * thermostat_state[1] / (degrees*target_temperature) - np.float32(1.0)
            thermostat_state[0] += dt * omega2 * ke_deviation
            thermostat_state[1] = np.float32(0.)
        return 
        
    step_nvt = cuda.jit(device=gridsync)(step_nvt)                        
    update_thermostat_state = cuda.jit(device=gridsync)(update_thermostat_state)    
    
    if gridsync:

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
            grid.sync()
            update_thermostat_state(integrator_params, time)
            return
        return cuda.jit(device=gridsync)(step)
    
    else:
        
        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            step_nvt[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
            update_thermostat_state[1, (1, 1)](integrator_params, time)
            return
        return step

def setup_integrator_nvt(configuration, interactions, temperature_function, tau, dt, compute_plan, verbose=True):
    integrator_step = make_step_nvt(configuration, temperature_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan, verbose=verbose)

    dt = np.float32(dt)
    omega2 = np.float32(4.0*np.pi*np.pi/tau/tau)
    degrees = configuration.N*configuration.D - configuration.D
    thermostat_state = np.zeros(2, dtype=np.float32)
    d_thermostat_state = cuda.to_device(thermostat_state)
    integrator_params =  (dt, omega2, degrees,  d_thermostat_state)
    
    return integrate, integrator_params


def make_step_nvt_langevin(configuration, compute_plan, verbose=True):
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

        # Unpack indicies for vectors and scalars
    # for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    # @cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync)
    # @cuda.jit(device=gridsync)
    def step_nvt_langevin(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NVT timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]
            REF: https://arxiv.org/pdf/1303.7011.pdf
        """

        dt, temperature, alpha, rng_states = integrator_params

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
                # REF: https://arxiv.org/pdf/1303.7011.pdf sec. 2.C.
                rng = xoroshiro128p_normal_float32(rng_states, local_id)
                beta = numba.float32((2*alpha*temperature*dt)**0.5)*rng
                numerator = 1 - alpha * dt / 2 / my_m
                denominator = 1 + alpha * dt / 2 / my_m
                a = numba.float32(numerator / denominator)
                b = numba.float32(1 / denominator)
                my_fsq += my_f[k] * my_f[k]
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  # Half step kinetic energy
                # Eq. (16) in https://arxiv.org/pdf/1303.7011.pdf
                my_v[k] = a * my_v[k] + b * my_f[k] / my_m * dt + b*beta/my_m
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
        return cuda.jit(device=gridsync)(step_nvt_langevin)  # return device function
    else:
        return cuda.jit(device=gridsync)(step_nvt_langevin)[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters

def test_step_langevin():

    import rumdpy as rp
    configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.85, T=0.8)
    configuration.copy_to_device()
    compute_plan = {'pb': 128, 'tp': 2, 'skin': 0.5, 'UtilizeNIII': False, 'gridsync': False}
    print('compute_plan: ', compute_plan)

    threads_per_block = 2
    blocks = 128
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=2023)

    pair_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(12,6))
    params = [[[4.0, -4.0, 2.5],], ]
    lennard_jones = rp.PairPotential(configuration, pair_potential, params=params, max_num_nbs=1000, compute_plan=compute_plan)
    pairs = lennard_jones.get_interactions(configuration, exclusions=None, compute_plan=compute_plan, verbose=True)
    integrator_step = make_step_nvt_langevin(configuration, compute_plan=compute_plan, verbose=True)
    integrate = make_integrator(configuration, integrator_step, pairs['interactions'], compute_plan=compute_plan, verbose=True)
    dt = np.float32(0.005)
    temperature = np.float32(1.20)
    alpha = np.float32(0.1)
    integrator_params = dt, temperature, alpha, rng_states
    inner_steps = 2
    outer_steps = 4

    scalars = []
    for i in range(outer_steps):
        integrate(configuration.d_vectors, configuration.d_scalars, configuration.d_ptype, configuration.d_r_im, configuration.simbox.d_data, pairs['interaction_params'], integrator_params, np.float32(0.0), inner_steps)
        scalars.append(np.sum(configuration.d_scalars.copy_to_host(), axis=0))
    print(scalars)