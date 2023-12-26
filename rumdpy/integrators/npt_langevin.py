""" 
NPT Langevin Leap-frog integrator
Based on N. Grønbech-Jensen and Oded Farago, J. Chem. Phys. 141, 194108 (2014). 
Ported from rumd.cu 
"""

import numpy as np
import numba
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from rumdpy.integrators.make_integrator import make_integrator, make_integrator_with_output

def make_step_npt_langevin(configuration, temperature_function, pressure_function, compute_plan, verbose=True):
    from numba.cuda.random import xoroshiro128p_normal_float32

    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating NPT langevin integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')
    
    # Unpack indicies for vectors and scalars
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    temperature_function = numba.njit(temperature_function)
    pressure_function = numba.njit(pressure_function)
    apply_PBC_dimension = numba.njit(configuration.simbox.apply_PBC_dimension)

    def copyParticleVirial(scalars, integrator_params):
        dt, alpha, alpha_baro, mass_baro, barostatModeISO, boxFlucCoord, rng_states, barostat_state, barostatVirial, length_ratio  = integrator_params
        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y
        my_w = scalars[global_id][w_id]

        #reset the barostatVirial to zero 
        if global_id == 0 and my_t == 0:
            barostatVirial[0] = numba.float32(0.0)
        
        cuda.syncthreads()
        
        if global_id < num_part and my_t == 0:
            cuda.atomic.add(barostatVirial, 0, my_w)  # factor of 6 already accounted for using virial_factor and virial_factor_NIII
        
        return

    def update_barostat_state(sim_box, integrator_params, time):
        dt, alpha, alpha_baro, mass_baro, barostatModeISO, boxFlucCoord, rng_states, barostat_state, barostatVirial, length_ratio = integrator_params
        temperature = temperature_function(time)
        pressure = pressure_function(time) 

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        if global_id == 0 and my_t == 0:

            #Copy barostat_state into current_barostat_state using a local aaray
            current_barostat_state = cuda.local.array(2, numba.float64)
            current_barostat_state[0] = barostat_state[0]
            current_barostat_state[1] = barostat_state[1]
            
            volume = sim_box[0]*sim_box[1]*sim_box[2]
            targetConfPressure = pressure - temperature * num_part / volume 
            barostatForce = barostatVirial[0] / volume - targetConfPressure

            random_number = xoroshiro128p_normal_float32(rng_states, 0)          # 0th random number state is reserved for barostat
            barostatRandomForce = math.sqrt(numba.float32(2.0) * alpha_baro * temperature * dt) * random_number 
          
            current_volume_velocity = current_barostat_state[1]
            inv_baro_mass = numba.float32(1.0) / mass_baro
            scaled_dt = numba.float32(0.5) * dt * alpha_baro * inv_baro_mass
            b_tilde = numba.float64(1.0) / (numba.float64(1.0) + scaled_dt)
            a_tilde = b_tilde * (numba.float64(1.0) - scaled_dt)

            new_volume_vel = a_tilde * current_volume_velocity + b_tilde * inv_baro_mass * (barostatForce * dt + barostatRandomForce)
            new_volume = volume + dt * new_volume_vel

            # Update the barostat state
            barostat_state[0] = new_volume / volume 
            barostat_state[1] = new_volume_vel

            # reset length_ratio to 1.0
            for i in range(3):
                length_ratio[i] = numba.float64(1.0)
            
            vol_scale_factor = barostat_state[0]
            lr_iso = math.pow(vol_scale_factor, numba.float64(1.0) / numba.float64(3.0))

            #update box length
            sim_box[0] += sim_box[0] * (lr_iso - 1.)
            sim_box[1] += sim_box[1] * (lr_iso - 1.)
            sim_box[2] += sim_box[2] * (lr_iso - 1.)
            
            # update length_ratio using cuda loop  
            for i in range(3):
                length_ratio[i] = lr_iso
                    
        return
    
    def step_npt_langevin(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NPT timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, 1)]
            REF: https://arxiv.org/pdf/1303.7011.pdf
        """

        dt, alpha, alpha_baro, mass_baro, barostatModeISO, boxFlucCoord, rng_states, barostat_state, barostatVirial, length_ratio = integrator_params
        temperature = temperature_function(time)
        pressure = pressure_function(time) 

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
                random_number = xoroshiro128p_normal_float32(rng_states, global_id + 1)  # +1 to avoid using the same random number state as the barostat
                beta = math.sqrt(numba.float32(2.0) * alpha * temperature * dt) * random_number
                  
                scaled_dt = numba.float32(0.5) * dt * alpha * numba.float32(1.0)/my_m
                prm_b = numba.float64(1.0) / (numba.float64(1.0) + scaled_dt)
                prm_a = prm_b * (numba.float64(1.0) - scaled_dt)
                
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  #  ke before
                my_fsq += my_f[k] * my_f[k] 

                my_v[k] = prm_a * my_v[k] + prm_b * (numba.float32(1.0)/my_m) * (my_f[k] * dt + beta)
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  #  ke after 
                
                L_factor = 2.*length_ratio[k] / (1. + length_ratio[k]) 
                my_r[k] = length_ratio[k] * my_r[k] + L_factor * my_v[k] * dt             
                
                # Apply PBC. 
                apply_PBC_dimension(my_r, r_im[global_id], sim_box, k)
           
            scalars[global_id][k_id] = numba.float32(0.5) * my_k
            scalars[global_id][fsq_id] = my_fsq

        return
    
    copyParticleVirial = cuda.jit(device=gridsync)(copyParticleVirial)
    update_barostat_state = cuda.jit(device=gridsync)(update_barostat_state)
    step_npt_langevin = cuda.jit(device=gridsync)(step_npt_langevin)

    if gridsync:                                              

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            copyParticleVirial(scalars, integrator_params)
            grid.sync()
            update_barostat_state(sim_box, integrator_params, time)
            grid.sync()
            step_npt_langevin(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
            return

        return cuda.jit(device=gridsync)(step)

    else:

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            copyParticleVirial[num_blocks, (pb, 1)](scalars, integrator_params)
            update_barostat_state[1, (1, 1)](sim_box, integrator_params, time)            
            step_npt_langevin[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
            return

        return step
    

def setup(configuration, interactions, temperature_function, pressure_function, alpha, alpha_baro, mass_baro,
           volume_velocity, barostatModeISO, boxFlucCoord, dt, seed, compute_plan, verbose=True):
   
    integrator_step = make_step_npt_langevin(configuration, temperature_function, pressure_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan, verbose=verbose) 
   
    rng_states = create_xoroshiro128p_states(configuration.N + 1, seed=seed)  # +1 for barostat dynamics 
    barostat_state = np.array([1.0, volume_velocity], dtype=np.float64)       # [0] = new_vol / old_vol , [1] = vol velocity
    d_barostat_state = cuda.to_device(barostat_state)
    barostatVirial = np.array([0.0], dtype=np.float32)
    d_barostatVirial = cuda.to_device(barostatVirial)
    d_length_ratio = cuda.to_device(np.ones(3, dtype=np.float32))  
    integrator_params = (np.float32(dt), np.float32(alpha), np.float32(alpha_baro), np.float32(mass_baro), 
                         barostatModeISO, np.int32(boxFlucCoord), rng_states, d_barostat_state, d_barostatVirial, d_length_ratio)  # Needs to be compatible with unpacking in
                                                                                                                                # step_npt_langevin()
    
    return integrate, integrator_params

def setup_output(configuration, interactions,  output_calculator, temperature_function, pressure_function, alpha, alpha_baro, mass_baro,
           volume_velocity, barostatModeISO, boxFlucCoord, dt, seed, compute_plan, verbose=True):
   
    integrator_step = make_step_npt_langevin(configuration, temperature_function, pressure_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator_with_output(configuration, integrator_step, interactions, output_calculator, compute_plan=compute_plan, verbose=verbose) 
   
    rng_states = create_xoroshiro128p_states(configuration.N + 1, seed=seed)  # +1 for barostat dynamics 
    barostat_state = np.array([1.0, volume_velocity], dtype=np.float64)       # [0] = new_vol / old_vol , [1] = vol velocity
    d_barostat_state = cuda.to_device(barostat_state)
    barostatVirial = np.array([0.0], dtype=np.float32)
    d_barostatVirial = cuda.to_device(barostatVirial)
    d_length_ratio = cuda.to_device(np.ones(3, dtype=np.float32))  
    integrator_params = (np.float32(dt), np.float32(alpha), np.float32(alpha_baro), np.float32(mass_baro), 
                         barostatModeISO, np.int32(boxFlucCoord), rng_states, d_barostat_state, d_barostatVirial, d_length_ratio)  # Needs to be compatible with unpacking in
                                                                                                                                # step_npt_langevin()
    
    return integrate, integrator_params
