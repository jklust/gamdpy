import numpy as np
import numba
import math
from numba import cuda

# rumdpy
import rumdpy as rp
from rumdpy.integrators import nve, nve_toxvaerd, nvt_nh, nvt_langevin, npt_langevin

# IO
import pandas as pd
import pickle
import sys
import h5py

class Simulation():
    def __init__(self, configuration, interactions, integrator, num_blocks, steps_per_block, 
                 compute_plan=None, storage='output.h5', output_manager='default', verbose=False):
                
        self.configuration = configuration
        if compute_plan==None:
            self.compute_plan = rp.get_default_compute_plan(self.configuration)

        self.interactions = interactions
        self.interactions_params = self.interactions.get_params(self.configuration, self.compute_plan, verbose)
        self.interactions_kernel = self.interactions.get_kernel(self.configuration, self.compute_plan, verbose)

        self.integrator = integrator
        self.integrator_params = self.integrator.get_params(self.configuration, verbose)
        self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)
        self.dt = self.integrator.dt
        
        self.num_blocks = num_blocks
        self.current_block = -1
        self.steps_per_block = steps_per_block
        self.storage = storage
         
        if output_manager=='default':
            self.steps_between_output = 16
            self.output_calculator = rp.make_scalar_calculator(self.configuration, self.steps_between_output, self.compute_plan)
            self.conf_saver = rp.make_conf_saver(self.configuration, self.compute_plan)
        elif output_manager==None or output_manager=='none':
            self.output_calculator = None
            self.conf_saver = None
        else:
            print('Did not understand output_manager = ', output_manager)

        # per block storage of configuration
        self.conf_per_block = int(math.log2(steps_per_block))+2 # Should be user controlable
        self.num_vectors = 2 # 'r' and 'r_im'
        print('Configurations per block (log2-storing):', self.conf_per_block)
        self.zero_conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
        self.d_conf_array = cuda.to_device(self.zero_conf_array)
        
        # per block storage of scalars
        self.num_scalars = 5
        self.scalar_saves_per_block = self.steps_per_block//self.steps_between_output
        self.zero_output_array = np.zeros((self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.zero_output_array) 
            
        if self.storage[-3:]=='.h5': # Saving in hdf5 format
            print('Saving results in hdf5 format. Filename:', self.storage)
            with h5py.File(self.storage, "w") as f:
                # Attributes for simulation (maybe save full configurations)
                f.attrs['dt'] = self.dt
                f.attrs['simbox_initial'] = self.configuration.simbox.lengths
                ds = f.create_dataset("ptype", shape=(self.configuration.N), dtype=np.int32)
                ds[:] = configuration.ptype
                #f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                #                chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32, compression="gzip")
                ds = f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D), 
                                    chunks=(1, 1, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
                ds.attrs['dummy'] = 1
                ds = f.create_dataset("scalars", shape=(self.num_blocks, self.scalar_saves_per_block, self.num_scalars), 
                                chunks=(1, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
                ds.attrs['steps_between_output'] = self.steps_between_output

        elif self.storage=='memory':
            # Should setup a dictionary, that exactly mirrors hdf5 file, so analysis programs can be the same 
            print(f'Storing results in memory. Expected footprint  {self.num_blocks*self.conf_per_block*self.num_vectors*self.configuration.N*self.configuration.D*4/1024/1024:.2f} MB.')
            # allocation delayed until beginning of run to let user reconsider
        else:
            print("WARNING: Results will not be stored. To change this use storage='filename.h5' or 'memory'")
            
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
                  
        self.integrate = self.make_integrator(self.configuration, self.integrator_kernel, self.interactions_kernel, self.output_calculator, self.conf_saver, self.compute_plan, True)
        
    def make_integrator(self, configuration, integration_step, compute_interactions, output_calculator, conf_saver, compute_plan, verbose=True ):
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
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, output_array, conf_array, time_zero, steps):
                time = time_zero
                for step in range(steps):
                    compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver != None:
                        conf_saver[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, conf_array, step)
                    time = time_zero + step*integrator_params[0]
                    integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if output_calculator != None:
                        output_calculator[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, output_array, step)
                if conf_saver != None:
                        conf_saver[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, conf_array, steps) # Save final configuration (if conditions fullfiled)
                return
            return integrator
        return            
            
    def run_blocks(self, num_blocks=-1):
        if num_blocks==-1:
            num_blocks=self.num_blocks
        self.last_num_blocks = num_blocks
        assert(num_blocks<=self.num_blocks) # Could be made OK with more blocks
        if self.storage=='memory':
            self.conf_blocks = np.zeros((self.num_blocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
        
        self.configuration.copy_to_device() 
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []
        
        start = cuda.event()
        end = cuda.event()
        zero = np.float32(0.0)
        start.record()
        
        for block in range(num_blocks):
            self.current_block = block
            self.d_output_array = cuda.to_device(self.zero_output_array) # Set output array to zero. Could probably be done faster
            self.integrate(self.configuration.d_vectors, 
                            self.configuration.d_scalars, 
                            self.configuration.d_ptype, 
                            self.configuration.d_r_im, 
                            self.configuration.simbox.d_data,       
                            self.interactions_params, 
                            self.integrator_params, 
                            self.d_output_array, 
                            self.d_conf_array, 
                            np.float32(block*self.steps_per_block*self.dt), 
                            self.steps_per_block)

            self.configuration.copy_to_host()
            self.vectors_list.append(self.configuration.vectors.copy())
            self.scalars_list.append(self.configuration.scalars.copy())
            self.simbox_data_list.append(self.configuration.simbox.lengths.copy()) # save to memory/hdf5
            #self.scalars_t.append(self.d_output_array.copy_to_host())     # save to memory/hdf5   
            
            if self.storage[-3:]=='.h5':
                with h5py.File(self.storage, "a") as f:
                    f['block'][block,:] = self.d_conf_array.copy_to_host()
                    f['scalars'][block,:] = self.d_output_array.copy_to_host()
            elif self.storage=='memory':
                self.conf_blocks[block] = self.d_conf_array.copy_to_host()
                
            #vol = (c1.simbox.lengths[0] * c1.simbox.lengths[1] * c1.simbox.lengths[2])
            #vol_t.append(vol)

            yield block
    
        # Finalizing run
        end.record()
        end.synchronize()
        print()
    
        self.timing_numba = cuda.event_elapsed_time(start, end)
        self.nbflag = self.interactions.nblist.d_nbflag.copy_to_host()    
        self.scalars_list = np.array(self.scalars_list)

    def print_status(self, per_particle=False):
        scalars = np.sum(self.configuration.scalars, axis=0)
        if per_particle:
            scalars /= self.configuration.N
        time = self.current_block * self.steps_per_block * self.dt
        print(f'\n{time= :<10.3f}', end=' ')
        for name in self.configuration.sid:
            idx = self.configuration.sid[name]
            print(f'{name}= {scalars[idx]:<10.3f}', end=' ')

        
    def print_summary(self):
        tps = self.last_num_blocks*self.steps_per_block/self.timing_numba*1000
        print('')
        print('steps :', self.last_num_blocks*self.steps_per_block)
        print('nbflag : ', self.nbflag)
        print('time :', self.timing_numba/1000, 's')
        print('TPS : ', tps )
            
