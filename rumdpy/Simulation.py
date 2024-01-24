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

class simulation():
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
                
            #vol = (c1.simbox.data[0] * c1.simbox.data[1] * c1.simbox.data[2])
            #vol_t.append(vol)

            if self.include_rdf:
                rdf_calculator(c1.d_vectors, c1.simbox.d_data, c1.d_ptype, pairs['interaction_params'], d_gr_bins)
                gr_bins.append(d_gr_bins.copy_to_host())
                d_gr_bins = cuda.to_device(gr_bins_zeros)
                
            # By default for now:
            self.conf.copy_to_host()
            self.vectors_list.append(self.conf.vectors.copy())
            self.scalars_list.append(self.conf.scalars.copy())
            self.simbox_data_list.append(self.conf.simbox.data.copy())

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
            