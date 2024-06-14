import numpy as np
import numba
import rumdpy as rp
from numba import cuda
import math

class SLLOD():
    def __init__(self, shear_rate, dt):
        self.shear_rate = shear_rate
        self.dt = dt
  
    def get_params(self, configuration, verbose=False):
        dt = np.float32(self.dt)
        sr_dt = np.float32(self.shear_rate*self.dt)
        return (dt,sr_dt)

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
        
        # JIT compile functions to be compiled into kernel
        apply_PBC = numba.njit(configuration.simbox.apply_PBC)
        update_box_shift = numba.njit(configuration.simbox.update_box_shift)
    
        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            """ Make one SLLOD timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, tp)]
            """
            
            # Unpack parameters. MUST be compatible with get_params() above
            dt,sr_dt = integrator_params

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
                
                if global_id == 0:
                    # Note: not synchronized!!!
                    update_box_shift(sim_box, sr_dt)
                
            return

        step = cuda.jit(device=gridsync)(step)

        if gridsync:
            return step  # return device function
        else:
            return step[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters
        
