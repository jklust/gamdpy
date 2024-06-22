import numpy as np
import numba
import rumdpy as rp
from numba import cuda
import math


## TO DO LIST FOR SLLOD (including LEBCs)
# 1. Check conservation of KE
# 2. Correct check of whether nb list needs to be built
# 3. Update images when box shift gets wrapped, or something equivalent

class SLLOD():
    def __init__(self, shear_rate, dt):
        self.shear_rate = shear_rate
        self.dt = dt
        
        # sum_pxpy, sum_pypy, sum_p2
        # three 'groups' of three sum variables
        self.thermostat_sums = np.zeros(9, dtype=np.float32) 
        self.d_thermostat_sums = cuda.to_device(self.thermostat_sums) 
        
  
    def get_params(self, configuration, verbose=False):
        dt = np.float32(self.dt)
        sr = np.float32(self.shear_rate)
        return (dt,sr, self.d_thermostat_sums)

    def get_kernel(self, configuration, compute_plan, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        if verbose:
            print(f'Generating SLLOD kernel for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id, k_id, fsq_id = [configuration.sid[key] for key in ['m', 'k', 'fsq']]     
        # was thinking that using a function oculd avoid synchronization
        # issues for updating the boxshift. But now I'm sure if it really
        # makes sense to use a function (the same way that NVT
        # does for temperature). There the temperature isn't stored anywhere.
        # Here I'm pretty sure the box_shift has to be stored together with the
        # other box details so interactions can always access it. So any
        # function has to update that one location and then we have to worry
        # about synchronization anyway
        #def strain_function(time):
        #    strain = self.shear_rate*time
 
        # JIT compile functions to be compiled into kernel
        apply_PBC = numba.njit(configuration.simbox.apply_PBC)
        update_box_shift = numba.njit(configuration.simbox.update_box_shift)
    
        def update_particle_data(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            """  Temporary update kernel, will be removed when SLLOD is properly implemented
            """
            
            # Unpack parameters. MUST be compatible with get_params() above
            dt,sr, thermostat_sums = integrator_params

            
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
                my_k = numba.float32(0.0)  # Kinetic energy
                my_fsq = numba.float32(0.0)  # force squared

                for k in range(D):
                    my_fsq += my_f[k] * my_f[k]
                    v_mean = numba.float32(0.0)

                    

                    v_mean += my_v[k]  # v(t-dt/2)
                    my_v[k] += my_f[k] / my_m * dt
                    v_mean += my_v[k]  # v(t+dt/2)
                    v_mean /= numba.float32(2.0)  # v(t) = (v(t-dt/2) + v(t+dt/2))/2
                    
                   

                    #  Basic: square the mean velocity
                    my_k += numba.float32(0.5) * my_m * v_mean * v_mean

                    my_r[k] += my_v[k] * dt
                
                apply_PBC(my_r, r_im[global_id], sim_box)

                
                scalars[global_id][k_id] = my_k
                scalars[global_id][fsq_id] = my_fsq
                
            return

        def call_update_box_shift(sim_box, integrator_params):
            dt, sr, thermostat_sums = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                sr_dt = sr * dt
                update_box_shift(sim_box, sr_dt)

        def initialize_g_factor(grid, vectors, scalars, integrator_params):
            dt, sr, thermostat_sums = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_m = scalars[global_id][m_id]
                
                
                my_pxpy = my_v[0] * my_v[1] * my_m
                my_pypy = my_v[1] * my_v[1] * my_m
                my_p2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k]**2
                my_p2 *= my_m
                
                # add to variables in 'group 0''
                cuda.atomic.add(thermostat_sums, 0, my_pxpy)
                cuda.atomic.add(thermostat_sums, 1, my_pypy)
                cuda.atomic.add(thermostat_sums, 2, my_p2)
                
                

        def integrate_sllod_b1(grid, vectors, scalars, integrator_params, time):
            dt, sr, thermostat_sums = integrator_params

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]

                # read sums from group 0
                sum_pxpy = thermostat_sums[0]
                sum_pypy = thermostat_sums[1]
                sum_p2 = thermostat_sums[2]
                # compute coefficients
                c1 = sr*sum_pxpy / sum_p2
                c2 = sr*sr * sum_pypy / sum_p2
                # g-factor for a half time-step
                g_factor = numba.float32(1.)/math.sqrt(numba.float32(1.) - c1*dt  + numba.float32(0.25) * c2 * dt**2)

                # update velocity 
                my_v[0] = g_factor * (my_v[0] - numba.float32(0.5)*sr*dt*my_v[1])
                for k in range(1, D):
                    my_v[k] *= g_factor

                # add to sums in group 1 needed for step B2
                my_p2 = numba.float32(0.)
                my_fp = numba.float32(0.)
                my_f2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k] * my_v[k] / my_m
                    my_fp += my_f[k] * my_v[k]
                    my_f2 += my_f[k] * my_f[k] / my_m
                cuda.atomic.add(thermostat_sums, 3, my_p2)
                cuda.atomic.add(thermostat_sums, 4, my_fp)
                cuda.atomic.add(thermostat_sums, 5, my_f2)

            # and reset group 2 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[6] = 0.
                thermostat_sums[7] = 0.
                thermostat_sums[8] = 0.


        def integrate_sllod_b2(grid, vectors, scalars, integrator_params, time):
            dt,sr, thermostat_sums = integrator_params
            
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]

                # read sums from group 1
                sum_p2 = thermostat_sums[3]
                sum_fp = thermostat_sums[4]
                sum_f2 = thermostat_sums[5]
                
                # compute coefficients
                alpha = sum_fp / sum_p2
                beta = math.sqrt(sum_f2 / sum_p2)
                h = (alpha + beta) / (alpha - beta)
                e = math.exp(-beta * dt)
                one = numba.float32(1.0)
                integrate_coefficient1 = (one - h) / (e - h/e)
                integrate_coefficient2 = (one + h - e - h/e)/((one-h)*beta)
                # update velocity
                
                for k in range(D):
                    my_v[k] = integrate_coefficient1 * (my_v[k] + integrate_coefficient2 * my_f[k] / my_m)

                # add to sums in group 2
                my_pxpy = my_v[0] * my_v[1] * my_m
                my_pypy = my_v[1] * my_v[1] * my_m
                my_p2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k]**2
                my_p2 *= my_m
                
                # add to variables in 'group 0''
                cuda.atomic.add(thermostat_sums, 6, my_pxpy)
                cuda.atomic.add(thermostat_sums, 7, my_pypy)
                cuda.atomic.add(thermostat_sums, 8, my_p2)

            # and reset group 0 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[0] = 0.
                thermostat_sums[1] = 0.
                thermostat_sums[2] = 0.


        def integrate_sllod_a_b1(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            dt,sr, thermostat_sums = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
            
                # read sums from group 2
                sum_pxpy = thermostat_sums[6]
                sum_pypy = thermostat_sums[7]
                sum_p2 = thermostat_sums[8]
                # compute coefficients
                c1 = sr*sum_pxpy / sum_p2
                c2 = sr*sr * sum_pypy / sum_p2
                # g-factor for a half time-step
                g_factor = numba.float32(1.)/math.sqrt(numba.float32(1.) - c1*dt  + numba.float32(0.25) * c2 * dt**2)
                
                
                # update velocity
                my_v[0] = g_factor * (my_v[0] - numba.float32(0.5)*sr*dt*my_v[1])
                for k in range(1, D):
                    my_v[k] *= g_factor
                
                # update position and apply bounday conditions
                my_r[0] += sr*dt*my_r[1] # rumd-3 has another term which seems to be incorrect (!)
                for k in range(k):
                    my_r[k] += my_v[k] * dt

                apply_PBC(my_r, r_im[global_id], sim_box)
                
                # add to sums in group 0 for next time integrate_B1 is called (at the next time step)
                my_pxpy = my_v[0] * my_v[1] * my_m
                my_pypy = my_v[1] * my_v[1] * my_m
                my_p2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k]**2
                my_p2 *= my_m
                
                cuda.atomic.add(thermostat_sums, 0, my_pxpy)
                cuda.atomic.add(thermostat_sums, 1, my_pypy)
                cuda.atomic.add(thermostat_sums, 2, my_p2)
            
            # and reset group 1 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[3] = 0.
                thermostat_sums[4] = 0.
                thermostat_sums[5] = 0.
        

                
        call_update_box_shift = cuda.jit(call_update_box_shift)
        update_particle_data = cuda.jit(device=gridsync)(update_particle_data)
        initialize_g_factor = cuda.jit(device=gridsync)(initialize_g_factor)
        integrate_sllod_b1 = cuda.jit(device=gridsync)(integrate_sllod_b1)
        integrate_sllod_b2 = cuda.jit(device=gridsync)(integrate_sllod_b2)
        integrate_sllod_a_b1 = cuda.jit(device=gridsync)(integrate_sllod_a_b1)


        if gridsync:
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time):

                
                if True:
                    
                    initialize_g_factor(grid, vectors, scalars, integrator_params) # should only be called the first time
                    grid.sync()
                    integrate_sllod_b1(grid, vectors, scalars, integrator_params, time)
                    grid.sync()
                    integrate_sllod_b2(grid, vectors, scalars, integrator_params, time)
                    grid.sync()
                    call_update_box_shift(sim_box, integrator_params)
                    # need to apply wrap to images!
                    # (alternatively store an extra integer with the box to count
                    # how many times it's been wrapped)
                    grid.sync()
                    integrate_sllod_a_b1(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                else:
                    # initial newtonian update, to be replaced by isokinetic sllod.
                    update_particle_data(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                    grid.sync()
                    call_update_box_shift(sim_box, integrator_params)
                return

            return cuda.jit(device=gridsync)(kernel)

        else:

            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
                update_particle_data[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                call_update_box_shift[1, (1, 1)](sim_box, integrator_params)
                return

        return kernel

