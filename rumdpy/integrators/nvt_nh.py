import numpy as np
import numba
import rumdpy as rp
from numba import cuda
import math
from rumdpy.integrators.make_integrator import make_integrator, make_integrator_with_output

def make_step_nvt(configuration, temperature_function, compute_plan, verbose=True, ):
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    D = configuration.D
    num_part = configuration.N
    num_blocks = (num_part - 1) // pb + 1

    if verbose:
        print(f'Generating NVT integrator for {num_part} particles in {D} dimensions:')
        print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
        print(f'\tNumber (virtual) particles: {num_blocks * pb}')
        print(f'\tNumber of threads {num_blocks * pb * tp}')

    # Unpack indices for vectors and scalars
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indices[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    temperature_function = numba.njit(temperature_function)
    # Could accept float and convert to function ourselves, to increase user friendliness
    apply_PBC = numba.njit(configuration.simbox.apply_PBC)

    # @cuda.jit('void(float32[:,:,:], float32[:,:], int32[:,:], float32[:], float32)', device=gridsync)
    # @cuda.jit(device=gridsync)
    def step_nvt(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
        """ Make one NVT timestep using Leap-frog
            Kernel configuration: [num_blocks, (pb, tp)]
        """

        dt, omega2, degrees, thermostat_state = integrator_params  # Put more in thermostat_state?

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        factor = np.float32(0.5) * thermostat_state[0] * dt
        plus = np.float32(1.) / (np.float32(1.) + factor)  # Possibly change to exp(...)
        minus = np.float32(1.) - factor  # Possibly change to exp(...)

        if global_id < num_part and my_t == 0:
            my_r = vectors[r_id][global_id]
            my_v = vectors[v_id][global_id]
            my_f = vectors[f_id][global_id]
            my_m = scalars[global_id][m_id]
            my_k = numba.float32(0.0)  # Kinetic energy
            my_fsq = numba.float32(0.0)  # force squared

            for k in range(D):
                my_fsq += my_f[k] * my_f[k]
                my_v[k] = plus * (minus * my_v[k] + my_f[k] / my_m * dt)
                my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]
                my_r[k] += my_v[k] * dt
                
            apply_PBC(my_r, r_im[global_id], sim_box)

            cuda.atomic.add(thermostat_state, 1, my_k)  # Probably slow! Not really
            scalars[global_id][k_id] = my_k
            scalars[global_id][fsq_id] = my_fsq
        return

    def update_thermostat_state(integrator_params, time):
        dt, omega2, degrees, thermostat_state = integrator_params  # Put more in thermostat_state?
        # Some of these can be compiled in, but will be less flexible

        my_block = cuda.blockIdx.x
        local_id = cuda.threadIdx.x
        global_id = my_block * pb + local_id
        my_t = cuda.threadIdx.y

        if global_id == 0 and my_t == 0:
            target_temperature = temperature_function(time)
            #if time>99499.:
            #    print(time, target_temperature, dt)
            ke_deviation = np.float32(2.0) * thermostat_state[1] / (degrees * target_temperature) - np.float32(1.0)
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

# Move the seup to Simulation?

def setup(configuration, interactions, temperature_function, tau, dt, compute_plan, verbose=True):
    
    integrator_step = make_step_nvt(configuration, temperature_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator(configuration, integrator_step, interactions, compute_plan=compute_plan,
                                verbose=verbose)

    dt = np.float32(dt)
    omega2 = np.float32(4.0 * np.pi * np.pi / tau / tau)
    degrees = configuration.N * configuration.D - configuration.D
    thermostat_state = np.zeros(2, dtype=np.float32)
    d_thermostat_state = cuda.to_device(thermostat_state)
    integrator_params = (dt, omega2, degrees, d_thermostat_state)   # Needs to be compatible with unpacking in
                                                                    # step_nvt(), and update_thermostat_state()
    return integrate, integrator_params

def setup_output(configuration, interactions, output_calculator, conf_saver, temperature_function, tau, dt, compute_plan, verbose=True):
    
    integrator_step = make_step_nvt(configuration, temperature_function, compute_plan=compute_plan, verbose=verbose)
    integrate = make_integrator_with_output(configuration, integrator_step, interactions, output_calculator, conf_saver, compute_plan=compute_plan,
                                verbose=verbose)

    dt = np.float32(dt)
    omega2 = np.float32(4.0 * np.pi * np.pi / tau / tau)
    degrees = configuration.N * configuration.D - configuration.D
    thermostat_state = np.zeros(2, dtype=np.float32)
    d_thermostat_state = cuda.to_device(thermostat_state)
    integrator_params = (dt, omega2, degrees, d_thermostat_state)   # Needs to be compatible with unpacking in
                                                                    # step_nvt(), and update_thermostat_state()
    return integrate, integrator_params

# Delay stuff in make_integrator_with_output to class Simulation
def setup_new(configuration, temperature, tau, dt, compute_plan, verbose=True):

    if not callable(temperature):
        temperature = rp.make_function_constant(value=float(temperature)) # better be a number...
    
    integrator_step = make_step_nvt(configuration, temperature, compute_plan=compute_plan, verbose=verbose)

    dt = np.float32(dt)
    omega2 = np.float32(4.0 * np.pi * np.pi / tau / tau)
    degrees = configuration.N * configuration.D - configuration.D
    thermostat_state = np.zeros(2, dtype=np.float32)
    d_thermostat_state = cuda.to_device(thermostat_state)
    integrator_params = (dt, omega2, degrees, d_thermostat_state)   # Needs to be compatible with unpacking in
                                                                    # step_nvt(), and update_thermostat_state()
    return integrator_step, integrator_params
