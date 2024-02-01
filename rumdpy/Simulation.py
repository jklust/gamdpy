import numpy as np
import numba
import math
from numba import cuda

# rumdpy
import rumdpy as rp
from rumdpy.integrators import nve, nve_toxvaerd, nvt, nvt_langevin, npt_langevin
#from analyze_LJ import analyze_LJ

# IO
import pandas as pd
import pickle
import sys
import h5py

class Simulation():
    def __init__(self, conf, integrator, dt, interactions, num_blocks, steps_per_block, steps_between_output, include_rdf=False, storage='hdf5', filename='output'):
        self.conf = conf
        self.integrator = integrator
        self.dt = dt
        self.interactions = interactions
        self.num_blocks = num_blocks
        self.steps_per_block = steps_per_block
        self.steps_between_output = steps_between_output
        self.include_rdf = include_rdf
        self.storage = storage
        self.filename = filename
        
        # per block storage of configuration
        self.conf_per_block = int(math.log2(steps_per_block))+2 # Should be user controlable
        self.num_vectors = 2 # 'r' and 'r_im'
        print('Configurations per block (log2-storing):', self.conf_per_block)
        self.zero_conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
        self.d_conf_array = cuda.to_device(self.zero_conf_array)
        
        # per block storage of scalars
        self.zero_output_array = np.zeros((self.steps_per_block//self.steps_between_output, 5), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.zero_output_array) 
            
        if self.storage=='hdf5':
            print('Saving results in hdf5 format. Filename:', filename+'.h5')
            with h5py.File(filename+'.h5', "w", libver="latest") as f:
                #dset = f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                #                        chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32, compression="gzip")
                dset = f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                                        chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
        elif self.storage=='memory':
            print(f'Storing results in memory. Expected footprint {self.num_blocks*self.conf_per_block*self.num_vectors*self.conf.N*self.conf.D*4/1024/1024:.2f} MB.')
            # allocation delayed until beginning of run to let user reconsider
        else:
            print("WARNING: Results will not be stored. To change this use storage='hdf5' or 'memory'")
            
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []
            
    def run(self, num_blocks=-1):
        if num_blocks==-1:
            num_blocks=self.num_blocks
        self.last_num_blocks = num_blocks
        assert(num_blocks<=self.num_blocks) # Could be made OK with more blocks
        if self.storage=='memory':
            self.conf_blocks = np.zeros((self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
        
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
            self.integrator[0](self.conf.d_vectors, self.conf.d_scalars, self.conf.d_ptype, self.conf.d_r_im, self.conf.simbox.d_data,  
                                 self.interactions['interaction_params'], self.integrator[1], 
                                 self.d_output_array, self.d_conf_array, 
                                 np.float32(block*self.steps_per_block*self.dt), self.steps_per_block)
        
            self.scalars_t.append(self.d_output_array.copy_to_host())        
            
            if self.storage=='hdf5':
                with h5py.File(self.filename+".h5", "a") as f:
                    f['block'][block,:] = self.d_conf_array.copy_to_host()
            elif self.storage=='memory':
                self.conf_blocks[block] = self.d_conf_array.copy_to_host()
                
            #vol = (c1.simbox.lengths[0] * c1.simbox.lengths[1] * c1.simbox.lengths[2])
            #vol_t.append(vol)

            if self.include_rdf:
                rdf_calculator(c1.d_vectors, c1.simbox.d_data, c1.d_ptype, pairs['interaction_params'], d_gr_bins)
                gr_bins.append(d_gr_bins.copy_to_host())
                d_gr_bins = cuda.to_device(gr_bins_zeros)
                
            # By default for now:
            self.conf.copy_to_host()
            self.vectors_list.append(self.conf.vectors.copy())
            self.scalars_list.append(self.conf.scalars.copy())
            self.simbox_data_list.append(self.conf.simbox.lengths.copy())

            yield block
    
        # Finalizing run
        end.record()
        end.synchronize()
    
        self.timing_numba = cuda.event_elapsed_time(start, end)
        #self.nbflag = LJ.nblist.d_nbflag.copy_to_host()    
        self.nbflag = self.interactions['interaction_params'][4].copy_to_host() # HACK!
        
        self.scalars_list = np.array(self.scalars_list)
        
    def print_summary(self):
        tps = self.last_num_blocks*self.steps_per_block/self.timing_numba*1000
        print('')
        print('steps :', self.last_num_blocks*self.steps_per_block)
        print('nbflag : ', self.nbflag)
        print('time :', self.timing_numba/1000, 's')
        print('TPS : ', tps )
            

class Simulation_new():
    def __init__(self, conf, interactions, integrator_step, num_blocks, steps_per_block, 
                 compute_plan=None, storage='output.h5', output_manager='default'):
        
# output_calculator, steps_between_output, conf_saver

        self.conf = conf
        self.integrator_step = integrator_step[0]
        self.integrator_params = integrator_step[1]
        
        self.dt = integrator_step[1][0] # Should be less cryptic
        self.interactions = interactions
        self.num_blocks = num_blocks
        self.current_block = -1
        self.steps_per_block = steps_per_block
        self.storage = storage
        self.compute_plan = compute_plan
        if compute_plan==None:
            self.compute_plan = rp.get_default_compute_plan(self.conf)
        
        if output_manager=='default':
            self.steps_between_output = 16
            self.output_calculator = rp.make_scalar_calculator(self.conf, self.steps_between_output, self.compute_plan)
            self.conf_saver = rp.make_conf_saver(self.conf, self.compute_plan)
        elif output_manager==None or output_manager=='none':
            self.output_calculator = None
            self.conf_saver = None
        else:
            print('Did not understand output_manager = ', output_manager)


        # per block storage of configuration
        self.conf_per_block = int(math.log2(steps_per_block))+2 # Should be user controlable
        self.num_vectors = 2 # 'r' and 'r_im'
        print('Configurations per block (log2-storing):', self.conf_per_block)
        self.zero_conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
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
                f.attrs['simbox_initial'] = self.conf.simbox.lengths
                ds = f.create_dataset("ptype", shape=(self.conf.N), dtype=np.int32)
                ds[:] = conf.ptype
                #f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                #                chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32, compression="gzip")
                ds = f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                                    chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
                ds.attrs['dummy'] = 1
                ds = f.create_dataset("scalars", shape=(self.num_blocks, self.scalar_saves_per_block, self.num_scalars), 
                                chunks=(1, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
                ds.attrs['steps_between_output'] = self.steps_between_output

        elif self.storage=='memory':
            # Should setup a dictionary, that exactly mirrors hdf5 file, so analysis programs can be the same 
            print(f'Storing results in memory. Expected footprint  {self.num_blocks*self.conf_per_block*self.num_vectors*self.conf.N*self.conf.D*4/1024/1024:.2f} MB.')
            # allocation delayed until beginning of run to let user reconsider
        else:
            print("WARNING: Results will not be stored. To change this use storage='filename.h5' or 'memory'")
            
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
                  
        self.integrator = self.make_integrator_with_output(self.conf, self.integrator_step, self.interactions['interactions'], self.output_calculator, self.conf_saver, self.compute_plan, True)
        
    def make_integrator_with_output(self, configuration, integration_step, compute_interactions, output_calculator, conf_saver, compute_plan, verbose=True ):
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
            
    def run_blocks(self, num_blocks=-1):
        if num_blocks==-1:
            num_blocks=self.num_blocks
        self.last_num_blocks = num_blocks
        assert(num_blocks<=self.num_blocks) # Could be made OK with more blocks
        if self.storage=='memory':
            self.conf_blocks = np.zeros((self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32)
        
        self.conf.copy_to_device() 
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
            self.integrator(self.conf.d_vectors, self.conf.d_scalars, self.conf.d_ptype, self.conf.d_r_im, self.conf.simbox.d_data,  
                                 self.interactions['interaction_params'], self.integrator_params, 
                                 self.d_output_array, self.d_conf_array, 
                                 np.float32(block*self.steps_per_block*self.dt), self.steps_per_block)


            self.conf.copy_to_host()
            self.vectors_list.append(self.conf.vectors.copy())
            self.scalars_list.append(self.conf.scalars.copy())
            self.simbox_data_list.append(self.conf.simbox.lengths.copy()) # save to memory/hdf5
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
        #self.nbflag = LJ.nblist.d_nbflag.copy_to_host()    
        self.nbflag = self.interactions['interaction_params'][4].copy_to_host() # HACK!
        
        self.scalars_list = np.array(self.scalars_list)

    def print_status(self, per_particle=False):
        scalars = np.sum(self.conf.scalars, axis=0)
        if per_particle:
            scalars /= self.conf.N
        time = self.current_block * self.steps_per_block * self.dt
        print(f'\n{time= :<10.3f}', end=' ')
        for name in self.conf.sid:
            idx = self.conf.sid[name]
            print(f'{name}= {scalars[idx]:<10.3f}', end=' ')

        
    def print_summary(self):
        tps = self.last_num_blocks*self.steps_per_block/self.timing_numba*1000
        print('')
        print('steps :', self.last_num_blocks*self.steps_per_block)
        print('nbflag : ', self.nbflag)
        print('time :', self.timing_numba/1000, 's')
        print('TPS : ', tps )
            
