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
    def __init__(self, configuration, interactions, integrator, num_steps=0, num_blocks=0, steps_per_block=0, 
                 compute_plan=None, storage='output.h5', scalar_output='default', conf_output='default', runtime_action='default', compute_stresses = False, verbose=False):
                
        self.configuration = configuration
        if compute_plan==None:
            self.compute_plan = rp.get_default_compute_plan(self.configuration)
        else:
            self.compute_plan = compute_plan

        self.compute_stresses = compute_stresses
        if self.compute_stresses and not configuration.compute_stresses:
           raise ValueError("Configuration must have compute_stresses set as well!")

        self.interactions = interactions
        self.interactions_params = self.interactions.get_params(self.configuration, self.compute_plan, verbose)
        self.interactions_kernel = self.interactions.get_kernel(self.configuration, self.compute_plan, self.compute_stresses, verbose)

        self.integrator = integrator
        self.integrator_params = self.integrator.get_params(self.configuration, verbose)
        self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)
        self.dt = self.integrator.dt
        
        
        
        if num_blocks==0:
            num_blocks = 32
            steps_per_block = 2**int( math.log2( math.ceil(num_steps / num_blocks )))
            num_blocks = math.ceil(num_steps / steps_per_block) 
            print('num_steps: ', num_steps)
            print('num_blocks: ', num_blocks)
            print('steps_per_block: ', steps_per_block)
             
        self.num_blocks = num_blocks
        self.current_block = -1
        self.steps_per_block = steps_per_block
        self.storage = storage
         
        if scalar_output == 'default':
            scalar_output = 16
        if type(scalar_output)==int and scalar_output>0:
            self.steps_between_output = scalar_output
            self.output_calculator = rp.make_scalar_calculator(self.configuration, self.steps_between_output, self.compute_plan)
        elif scalar_output==None or scalar_output =='none' or scalar_output < 1:
            self.output_calculator = None
            self.steps_between_output = scalar_output
        else:
            print('Did not understand scalar_output = ', scalar_output)

        if conf_output=='default':
            self.conf_saver = rp.make_conf_saver(self.configuration, self.compute_plan)
        elif conf_output==None or conf_output=='none':
            self.conf_saver = None
        else:
            print('Did not understand conf_output = ', conf_output)

        if runtime_action == 'default':
            runtime_action = 100
        if type(runtime_action)==int and runtime_action>0:
            self.steps_between_runtime_action = runtime_action
            self.runtime_action_executor = rp.make_runtime_action_executor(self.configuration, self.steps_between_runtime_action, self.compute_plan)
        elif runtime_action == None or runtime_action == 'none' or runtime_action <= 0:
            self.runtime_action_executor = None
            #self.steps_between_runtime_action = 0
        else:
            print('Did not understand runtime_action = ', runtime_action)


        # per block storage of configuration
        if self.conf_saver != None:
            self.conf_per_block = int(math.log2(steps_per_block))+2 # Should be user controlable
            if verbose:
                print('Configurations per block (log2-storing):', self.conf_per_block)
        else:
            self.conf_per_block = 1
        self.num_vectors = 2 # 'r' and 'r_im'
        self.zero_conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
        self.d_conf_array = cuda.to_device(self.zero_conf_array)
        
        # per block storage of scalars
        self.num_scalars = 6
        #include CM velocity
        self.num_scalars += self.configuration.D

        if self.output_calculator != None:
            self.scalar_saves_per_block = self.steps_per_block//self.steps_between_output
        else:
            self.scalar_saves_per_block = 1

        self.zero_output_array = np.zeros((self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.zero_output_array) 
        
        # Storage for momentum resetting
        self.cm_velocity = np.zeros(self.configuration.D+1, dtype=np.float32) # Total mass summed in last index of cm_velocity
        self.d_cm_velocity = cuda.to_device(self.cm_velocity)
            
        if self.storage[-3:]=='.h5': # Saving in hdf5 format
            if verbose:
                print('Saving results in hdf5 format. Filename:', self.storage)
            with h5py.File(self.storage, "w") as f:
                # Attributes for simulation (maybe save full configurations)
                f.attrs['dt'] = self.dt
                f.attrs['simbox_initial'] = self.configuration.simbox.lengths
                ds = f.create_dataset("ptype", shape=(self.configuration.N), dtype=np.int32)
                ds[:] = configuration.ptype
                #f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                #                chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32, compression="gzip")
                if self.conf_saver != None:
                    ds = f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D),
                                          chunks=(1, 1, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)

                if self.output_calculator != None:
                    ds = f.create_dataset("scalars", shape=(self.num_blocks, self.scalar_saves_per_block, self.num_scalars),
                                          chunks=(1, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
                    f.attrs['steps_between_output'] = self.steps_between_output

        elif self.storage=='memory':
            # Setup a dictionary that exactly mirrors hdf5 file, so analysis programs can be the same
            self.output = {}
            self.output['attrs'] = {'dt':self.dt, 'simbox_initial':self.configuration.simbox.lengths.copy() }
            self.output['ptype'] = configuration.ptype.copy()
            if self.conf_saver != None:
                self.output['block'] = 0
            print(f'Storing results in memory. Expected footprint  {self.num_blocks*self.conf_per_block*self.num_vectors*self.configuration.N*self.configuration.D*4/1024/1024:.2f} MB.')
            # allocation delayed until beginning of run to let user reconsider
            if self.output_calculator != None:
                self.output['scalars'] = 0
                self.output['attrs']['steps_between_output'] = self.steps_between_output
        else:
            print("WARNING: Results will not be stored. To change this use storage='filename.h5' or 'memory'")
            
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
                  
        self.integrate = self.make_integrator(self.configuration, self.integrator_kernel, self.interactions_kernel,
                                              self.output_calculator, self.conf_saver, self.runtime_action_executor, self.compute_plan, True)
        
    def make_integrator(self, configuration, integration_step, compute_interactions, output_calculator, conf_saver, runtime_action_executor, compute_plan, verbose=True ):
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
        #if runtime_action_executor != None:
        #    runtime_action_executor = cuda.jit(device=gridsync)(numba.njit(runtime_action_executor))

            
        if gridsync:
            # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
            @cuda.jit
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, output_array, conf_array, cm_velocity, time_zero, steps):
                grid = cuda.cg.this_grid()
                time = time_zero
                for step in range(steps):
                    compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver != None:
                        conf_saver(grid, vectors, scalars, r_im, sim_box, conf_array, step)
                    grid.sync()
                    time = time_zero + step*integrator_params[0]
                    integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if runtime_action_executor != None:
                        runtime_action_executor(grid, vectors, scalars, r_im, sim_box, step, cm_velocity)

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
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, output_array, conf_array, cm_velocity, time_zero, steps):
                time = time_zero
                for step in range(steps):
                    compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver != None:
                        conf_saver[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, conf_array, step)
                    time = time_zero + step*integrator_params[0]
                    integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if output_calculator != None:
                        output_calculator[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, output_array, step)
                    if runtime_action_executor != None:
                        runtime_action_executor(0, vectors, scalars, r_im, sim_box, step, cm_velocity)

                        
                if conf_saver != None:
                        conf_saver[num_blocks, (pb, 1)](0, vectors, scalars, r_im, sim_box, conf_array, steps) # Save final configuration (if conditions fullfiled)
                return
            return integrator
        return            

    # simple run function
    def run(self):
        for block in self.blocks():
            print(self.status(per_particle=True))
        print(self.summary())

    # generator for running simulation one block at a time
    def blocks(self, num_blocks=-1):
        if num_blocks==-1:
            num_blocks=self.num_blocks
        self.last_num_blocks = num_blocks
        assert(num_blocks<=self.num_blocks) # Could be made OK with more blocks
        if self.storage=='memory':
            if self.conf_saver != None:
                self.output['block'] = np.zeros((self.num_blocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
            if self.output_calculator != None:
                self.output['scalars'] = np.zeros((self.num_blocks, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)

        self.configuration.copy_to_device()
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []
        
        start = cuda.event()
        end = cuda.event()
        start_block = cuda.event()
        end_block = cuda.event()
        zero = np.float32(0.0)
        block_times = []

        start.record()    
        for block in range(num_blocks):
            start_block.record()
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
                            self.d_cm_velocity,
                            np.float32(block*self.steps_per_block*self.dt), 
                            self.steps_per_block)

            self.configuration.copy_to_host()
            self.vectors_list.append(self.configuration.vectors.copy())
            self.scalars_list.append(self.configuration.scalars.copy())
            self.simbox_data_list.append(self.configuration.simbox.lengths.copy()) # save to memory/hdf5
            self.scalars_t.append(self.d_output_array.copy_to_host())     # save to memory/hdf5   
            
            if self.storage[-3:]=='.h5':
                with h5py.File(self.storage, "a") as f:
                    if self.conf_saver != None:
                        f['block'][block,:] = self.d_conf_array.copy_to_host()
                    if self.output_calculator != None:
                        f['scalars'][block,:] = self.d_output_array.copy_to_host()
            elif self.storage=='memory':
                if self.conf_saver != None:
                    self.output['block'][block,:] = self.d_conf_array.copy_to_host()
                if self.output_calculator != None:
                    self.output['scalars'][block,:] = self.d_output_array.copy_to_host()
                
            #vol = (c1.simbox.lengths[0] * c1.simbox.lengths[1] * c1.simbox.lengths[2])
            #vol_t.append(vol)
            end_block.record()
            end_block.synchronize()
            block_times.append(cuda.event_elapsed_time(start_block, end_block))
            yield block
    
        # Finalizing run
        end.record()
        end.synchronize()
        #print()
    
        self.timing_numba = cuda.event_elapsed_time(start, end)
        self.timing_numba_blocks = np.array(block_times)
        self.nbflag = self.interactions.nblist.d_nbflag.copy_to_host()    
        self.scalars_list = np.array(self.scalars_list)

    def status(self, per_particle=False):
        time = self.current_block * self.steps_per_block * self.dt
        st = f'{time= :<10.3f}'
        for name in self.configuration.sid:
            data = np.sum(self.configuration[name])
            if per_particle:
                data /=  self.configuration.N
            st += f'{name}= {data:<10.3f}'
        return st

    def summary(self):
        time_total = self.timing_numba/1000
        tps_total = self.last_num_blocks*self.steps_per_block/time_total
        time_sim = np.sum(self.timing_numba_blocks)/1000
        tps_sim = self.last_num_blocks*self.steps_per_block/time_sim

        if self.timing_numba_blocks.shape[0]>1:
            extratime_firstblock = (self.timing_numba_blocks[0] 
                                    - np.mean(self.timing_numba_blocks[1:]))/1000
            time_sim_minus_extra = time_sim - extratime_firstblock
            tps_sim_minus_extra = self.last_num_blocks*self.steps_per_block/time_sim_minus_extra

        st  = f'Particles : {self.configuration.N} \n'
        st += f'Steps : {self.last_num_blocks*self.steps_per_block} \n'
        st += f'nbflag : {self.nbflag} \n'
        st += f'Total time (incl. time spent between blocks): {time_sim:.2f} s \n'
        st += f'Simulation time : {time_total:.2f} s \n'
        st += f'Extra time 1.st block (presumably JIT): {extratime_firstblock:.2f} s \n'
        st += f'TPS_total : {tps_total:.2e} \n'
        st += f'TPS_sim : {tps_sim:.2e} \n'
        if self.timing_numba_blocks.shape[0]>1:
            st += f'TPS_sim_minus_extra : {tps_sim_minus_extra:.2e} \n'    
        return st
