import numpy as np
import numba
import math
from numba import cuda

# rumdpy
import rumdpy as rp

# IO
import h5py

class Simulation():
    def __init__(self, configuration, interactions, integrator, num_steps=0, num_timeblocks=0, steps_per_timeblock=0,
                 compute_plan=None, storage='output.h5', scalar_output='default', conf_output='default', 
                 steps_between_momentum_reset='default', compute_stresses=False, verbose=False, timing=True):
                
        self.configuration = configuration
        if compute_plan==None:
            self.compute_plan = rp.get_default_compute_plan(self.configuration)
        else:
            self.compute_plan = compute_plan

        self.compute_stresses = compute_stresses
        if self.compute_stresses and not configuration.compute_stresses:
           raise ValueError("Configuration must have compute_stresses set as well!")

        if type(interactions) == list:
            self.interactions = interactions
        else:
            self.interactions = [interactions,]
        self.interactions_kernel, self.interactions_params = rp.add_interactions_list(self.configuration, self.interactions, 
                                                                                      compute_plan=self.compute_plan, 
                                                                                      compute_stresses=compute_stresses, 
                                                                                      verbose=verbose)

        self.integrator = integrator
        self.integrator_params = self.integrator.get_params(self.configuration, verbose)
        self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)
        self.dt = self.integrator.dt
        
        if num_timeblocks==0:
            num_timeblocks = 32
            steps_per_timeblock = 2 ** int(math.log2(math.ceil(num_steps / num_timeblocks)))
            num_timeblocks = math.ceil(num_steps / steps_per_timeblock)
            print('num_steps: ', num_steps)
            print('num_blocks: ', num_timeblocks)
            print('steps_per_block: ', steps_per_timeblock)
             
        self.num_blocks = num_timeblocks
        self.current_block = -1
        self.steps_per_block = steps_per_timeblock
        self.storage = storage
        self.timing = timing

        if self.storage[-3:]=='.h5': # Saving in hdf5 format
            with h5py.File(self.storage, "w") as f:
                # Attributes for simulation (maybe save full configurations)
                f.attrs['dt'] = self.dt
                f.attrs['simbox_initial'] = self.configuration.simbox.lengths
                ds = f.create_dataset("ptype", shape=(self.configuration.N), dtype=np.int32)
                ds[:] = configuration.ptype                            
        elif self.storage=='memory':
            # Setup a dictionary that exactly mirrors hdf5 file, so analysis programs can be the same
            self.output = {}
            self.output['attrs'] = {'dt':self.dt, 'simbox_initial':self.configuration.simbox.lengths.copy() }
            self.output['ptype'] = configuration.ptype.copy()
                   
        # Momentum reset
        if steps_between_momentum_reset>0:
            self.momentum_reset = rp.MomentumReset(steps_between_momentum_reset)
            self.momentum_reset_params = self.momentum_reset.get_params(self.configuration, self.compute_plan)
            self.momentum_reset_kernel = self.momentum_reset.get_kernel(self.configuration, self.compute_plan)
        else:
            self.momentum_reset_kernel = None
            self.momentum_reset_params = (0,)

        # Scalar saving
        if scalar_output == 'default':
            scalar_output = 16
        if scalar_output==None or scalar_output =='none' or scalar_output < 1:
            self.output_calculator = None
            self.output_calculator_kernel = None
            self.output_calculator_params = (0,)
        else:
            self.output_calculator = rp.ScalarSaver(configuration, scalar_output, num_timeblocks, steps_per_timeblock, storage)
            if self.storage[-3:]!='.h5':
                self.output.update(self.output_calculator.output)
            self.output_calculator_params = self.output_calculator.get_params(self.configuration, self.compute_plan)
            self.output_calculator_kernel = self.output_calculator.get_kernel(self.configuration, self.compute_plan)
            
        # Saving of configurations
        if conf_output=='default':
            self.conf_saver = rp.ConfSaver(self.configuration, num_timeblocks, steps_per_timeblock, storage)
            if self.storage[-3:]!='.h5':
                self.output.update(self.conf_saver.output)
            self.conf_saver_kernel = self.conf_saver.get_kernel(self.configuration, self.compute_plan)
            self.conf_saver_params = self.conf_saver.get_params(self.configuration, self.compute_plan)
        elif conf_output==None or conf_output=='none':
            self.conf_saver = None
            self.conf_saver_kernel = None
            self.conf_saver_params = (0,)
        else:
            print('Did not understand conf_output = ', conf_output)

        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
                  
        self.integrate = self.make_integrator(self.configuration, self.integrator_kernel, self.interactions_kernel,
                                              self.output_calculator_kernel, self.conf_saver_kernel, self.momentum_reset_kernel, 
                                              self.compute_plan, True)
        
    def make_integrator(self, configuration, integration_step, compute_interactions, output_calculator_kernel, conf_saver_kernel, momentum_reset_kernel, compute_plan, verbose=True ):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1
    
        if gridsync:
            # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
            @cuda.jit
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, conf_saver_params, momentum_reset_params, output_calculator_params, time_zero, steps):
                grid = cuda.cg.this_grid()
                for step in range(steps):
                    compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver_kernel != None:
                        conf_saver_kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params)
                    grid.sync()
                    time = time_zero + step*integrator_params[0]
                    integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if momentum_reset_kernel != None:
                        momentum_reset_kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params)
                    if output_calculator_kernel != None:
                        output_calculator_kernel(grid, vectors, scalars, r_im, sim_box, step, output_calculator_params)
                    
                    grid.sync()
                if conf_saver_kernel != None:
                    conf_saver_kernel(grid, vectors, scalars, r_im, sim_box, steps, conf_saver_params) # Save final configuration (if conditions fullfiled)
                return

            return integrator[num_blocks, (pb, tp)]

        else:

            # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params, conf_saver_params, momentum_reset_params, output_calculator_params, time_zero, steps):
                for step in range(steps):
                    compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                    if conf_saver_kernel != None:
                        conf_saver_kernel(0, vectors, scalars, r_im, sim_box, step, conf_saver_params)
                    time = time_zero + step*integrator_params[0]
                    integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time)
                    if output_calculator_kernel != None:
                        output_calculator_kernel(0, vectors, scalars, r_im, sim_box, step, output_calculator_params)
                    if momentum_reset_kernel != None:
                        momentum_reset_kernel(0, vectors, scalars, r_im, sim_box, step, momentum_reset_params)
                     
                if conf_saver_kernel != None:
                    conf_saver_kernel(0, vectors, scalars, r_im, sim_box, steps, conf_saver_params) # Save final configuration (if conditions fullfiled)
                return
            return integrator
        return            

    # simple run function
    def run(self):
        for _ in self.timeblocks():
            print(self.status(per_particle=True))
        print(self.summary())

    # generator for running simulation one block at a time
    def timeblocks(self, num_blocks=-1):
        if num_blocks==-1:
            num_blocks=self.num_blocks
        self.last_num_blocks = num_blocks
        assert(num_blocks<=self.num_blocks) # Could be made OK with more blocks
               
        self.configuration.copy_to_device()
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []
       
        if self.timing:
            start = cuda.event()
            end = cuda.event()
            start_block = cuda.event()
            end_block = cuda.event()
            block_times = []

            start.record()    

        zero = np.float32(0.0)

        for block in range(num_blocks):
            if self.timing: start_block.record()
            self.current_block = block
            #self.d_output_array = cuda.to_device(self.zero_output_array) # Set output array to zero. Could probably be done faster
            self.integrate(self.configuration.d_vectors, 
                            self.configuration.d_scalars, 
                            self.configuration.d_ptype, 
                            self.configuration.d_r_im, 
                            self.configuration.simbox.d_data,       
                            self.interactions_params, 
                            self.integrator_params, 
                            self.conf_saver_params, 
                            self.momentum_reset_params,
                            self.output_calculator_params,
                            np.float32(block*self.steps_per_block*self.dt), 
                            self.steps_per_block)

            self.configuration.copy_to_host()
            self.vectors_list.append(self.configuration.vectors.copy()) # Needed for 3D viz, should use memory/hdf5
            self.scalars_list.append(self.configuration.scalars.copy())            # same
            self.simbox_data_list.append(self.configuration.simbox.lengths.copy()) # same
            #self.scalars_t.append(self.d_output_array.copy_to_host())              # same
                
            if self.output_calculator != None:
                self.output_calculator.update_at_end_of_timeblock(block)
            if self.conf_saver != None:
                self.conf_saver.update_at_end_of_timeblock(block)

            if self.timing:
                end_block.record()
                end_block.synchronize()
                block_times.append(cuda.event_elapsed_time(start_block, end_block))
            yield block
    
        # Finalizing run
        if self.timing:
            end.record()
            end.synchronize()
    
            self.timing_numba = cuda.event_elapsed_time(start, end)
            self.timing_numba_blocks = np.array(block_times)
        self.nbflag = self.interactions[0].nblist.d_nbflag.copy_to_host()    
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
        if self.timing:
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
        if self.timing:
            st += f'Total time (incl. time spent between blocks): {time_total:.2f} s \n'
            st += f'Simulation time : {time_sim:.2f} s \n'
            st += f'Extra time 1.st block (presumably JIT): {extratime_firstblock:.2f} s \n'
            st += f'TPS_total : {tps_total:.2e} \n'
            st += f'TPS_sim : {tps_sim:.2e} \n'
            if self.timing_numba_blocks.shape[0]>1:
                st += f'TPS_sim_minus_extra : {tps_sim_minus_extra:.2e} \n'    
        return st
