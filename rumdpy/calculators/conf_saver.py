import numpy as np
import numba
import math
from numba import cuda, config

import h5py


class ConfSaver():
    """ Runtime_action for saving configurations during timeblock
        - for now only logarithmic saving
    """

    def __init__(self, configuration, num_timeblocks: int, steps_per_timeblock: int, storage: str, verbose=False) -> None:
        self.configuration = configuration

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks

        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        if storage != 'memory' and storage[-3:] != '.h5':  # Either in-memory or hdf5 (others could be added)
            raise ValueError(f'storage({storage}) needs to be "memory" or end at ".h5"')
        self.storage = storage

        self.conf_per_block = int(math.log2(self.steps_per_timeblock)) + 2  # Should be user controllable

        self.num_vectors = 2  # 'r' and 'r_im' (for now!)

        self.sid = {"r":0, "r_im":1}

        # Setup output
#        if self.storage[-3:] == '.h5':  # Saving in hdf5 format
#            with h5py.File(self.storage, "a") as f:
                #f.create_dataset("block", shape=(self.num_blocks, self.conf_per_block, self.num_vectors, self.conf.N, self.conf.D), 
                #                chunks=(1, 1, self.num_vectors, self.conf.N, self.conf.D), dtype=np.float32, compression="gzip")
#                ds = f.create_dataset("block", shape=(
#                self.num_timeblocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D),
#                                      chunks=(1, 1, self.num_vectors, self.configuration.N, self.configuration.D),
#                                      dtype=np.float32)
#                f.attrs['vectors_names'] = list(self.sid.keys())
#        elif self.storage == 'memory':
            # Setup a dictionary that exactly mirrors hdf5 file, so analysis programs can be the same
#            self.output = {}
#            self.output['block'] = np.zeros((self.num_timeblocks, self.conf_per_block, self.num_vectors,
#                                             self.configuration.N, self.configuration.D), dtype=np.float32)
            #self.output['attrs']['vectors_names'] = list(self.sid.keys()) #LC: at one point should be like this
#            self.output['vectors_names'] = list(self.sid.keys())
#            if verbose:
#                print(
#                    f'Storing results in memory. Expected footprint  {self.num_timeblocks * self.conf_per_block * self.num_vectors * self.configuration.N * self.configuration.D * 4 / 1024 / 1024:.2f} MB.')
#        else:
#            print("WARNING: Results will not be stored. To change this use storage='filename.h5' or 'memory'")
        with h5py.File(self.storage, "a") as f:
            ds = f.create_dataset("block", shape=(
                self.num_timeblocks, self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D),
                chunks=(1, 1, self.num_vectors, self.configuration.N, self.configuration.D), dtype=np.float32)
            f.attrs['vectors_names'] = list(self.sid.keys())

        flag = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        self.zero_kernel = self.make_zero_kernel()
        config.CUDA_LOW_OCCUPANCY_WARNINGS = flag

    def get_params(self, configuration, compute_plan):
        self.conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D),
                                   dtype=np.float32)
        self.d_conf_array = cuda.to_device(self.conf_array)
        return (self.d_conf_array,)

    def make_zero_kernel(self):
        # Unpack parameters from configuration and compute_plan
        D, num_part = self.configuration.D, self.configuration.N
        pb = 32
        num_blocks = (num_part - 1) // pb + 1

        def zero_kernel(array):
            Nx, Ny, Nz, Nw = array.shape
            global_id = cuda.grid(1)

            if global_id < Nz:  # particles
                for i in range(Nx):
                    for j in range(Ny):
                        for k in range(Nw):
                            array[i, j, global_id, k] = numba.float32(0.0)

        zero_kernel = cuda.jit(zero_kernel)
        return zero_kernel[num_blocks, pb]

    def update_at_end_of_timeblock(self, block: int):
        #if self.storage[-3:] == '.h5':
        #    with h5py.File(self.storage, "a") as f:
        #        f['block'][block, :] = self.d_conf_array.copy_to_host()
        #elif self.storage == 'memory':
        #    self.output['block'][block, :] = self.d_conf_array.copy_to_host()
        with h5py.File(self.storage, "a") as f:
            f['block'][block, :] = self.d_conf_array.copy_to_host()

        self.zero_kernel(self.d_conf_array)

    def get_kernel(self, configuration, compute_plan, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1

        # Unpack indices for scalars to be compiled into kernel  
        r_id, = [configuration.vectors.indices[key] for key in ['r', ]]

        def kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params):
            conf_array, = conf_saver_params

            Flag = False
            if step == 0:
                Flag = True
                save_index = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    Flag = True
                    save_index = b + 1

            if Flag:
                global_id, my_t = cuda.grid(2)
                if global_id < num_part and my_t == 0:
                    for k in range(D):
                        conf_array[save_index, 0, global_id, k] = vectors[r_id][global_id, k]
                        conf_array[save_index, 1, global_id, k] = np.float32(r_im[global_id, k])
            return

        kernel = cuda.jit(device=gridsync)(kernel)

        if gridsync:
            return kernel  # return device function
        else:
            return kernel[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters
