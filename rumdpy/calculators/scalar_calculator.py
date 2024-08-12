import numpy as np
import numba
import math
from numba import cuda, config
import h5py

class ScalarSaver():

    def __init__(self, configuration,
                 steps_between_output:int, num_timeblocks:int, steps_per_timeblock:int, storage:str) -> None:

        self.configuration = configuration

        if type(steps_between_output) != int or steps_between_output < 0:
            raise ValueError(f'steps_between_output ({steps_between_output}) should be non-negative integer.')
        self.steps_between_output = steps_between_output

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks

        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        if steps_between_output >= steps_per_timeblock:
            raise ValueError(f'scalar_output ({steps_between_output}) must be less than steps_per_timeblock ({steps_per_timeblock})')

        if storage != 'memory' and storage[-3:] != '.h5': # Either in-memory or hdf5 (others could be added)
            raise ValueError(f'storage({storage}) needs to be "memory" or end at ".h5"')
        self.storage = storage


        # per block storage of scalars
        self.num_scalars = 6
        self.num_scalars += self.configuration.D #include CM momentum
        self.num_scalars += 1 #include XY component of stress (temporary!!!)
        self.sid = {'U':0, 'W':1, 'lapU':2, 'Fsq':3, 'K':4, 'Vol':5, 'Px':6, 'Py':7, 'Pz':8, 'Sxy':9} # 3D for now!!!

        self.scalar_saves_per_block = self.steps_per_timeblock//self.steps_between_output

        # Setup output
        shape = (self.num_timeblocks, self.scalar_saves_per_block, self.num_scalars)
        #if self.storage[-3:]=='.h5': # Saving in hdf5 format
        #    with h5py.File(self.storage, 'a') as f:
        #        f.create_dataset('scalars', shape=shape,
        #                        chunks=(1, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
        #        f.attrs['steps_between_output'] = self.steps_between_output
        #        f.attrs['scalars_names'] = list(self.sid.keys())
        #elif self.storage=='memory': 
            # Setup a dictionary that mirrors hdf5 file, so analysis programs can be (almost) the same
        #    self.output = {}
        #    self.output['scalars'] = np.zeros(shape=shape, dtype=np.float32)
            #self.output['attrs']['steps_between_output'] = self.steps_between_output #LC: at one pint should be like this
            #self.output['attrs']['scalars_names'] = list(self.sid.keys())            #LC: at one pint should be like this
        #    self.output['steps_between_output'] = self.steps_between_output
        #    self.output['scalars_names'] = list(self.sid.keys())
        with h5py.File(self.storage, 'a') as f:
            f.create_dataset('scalars', shape=shape,
                    chunks=(1, self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
            f.attrs['steps_between_output'] = self.steps_between_output
            f.attrs['scalars_names'] = list(self.sid.keys())

        flag = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        self.zero_kernel = self.make_zero_kernel()
        config.CUDA_LOW_OCCUPANCY_WARNINGS = flag

    def make_zero_kernel(self):
        
        def zero_kernel(array):
            Nx, Ny = array.shape
            #i, j = cuda.grid(2) # doing simple 1 thread kernel for now ...
            for i in range(Nx):
                for j in range(Ny):
                    array[i,j] = numba.float32(0.0)

        zero_kernel = cuda.jit(zero_kernel)
        return zero_kernel[1,1]
     
    def get_params(self, configuration, compute_plan):
        
        self.output_array = np.zeros((self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.output_array)
        self.params = (self.steps_between_output, self.d_output_array)
        return self.params
    
    def initialize_before_timeblock(self):
        self.zero_kernel(self.d_output_array)

    def update_at_end_of_timeblock(self, block:int):
        #if self.storage[-3:]=='.h5':
        #    with h5py.File(self.storage, "a") as f:
        #        f['scalars'][block,:] = self.d_output_array.copy_to_host()
        #elif self.storage=='memory':
        #        self.output['scalars'][block,:] = self.d_output_array.copy_to_host()
        with h5py.File(self.storage, "a") as f:
            f['scalars'][block,:] = self.d_output_array.copy_to_host()
    
    def get_kernel(self, configuration, compute_plan):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1
        
        # Unpack indices for scalars to be compiled into kernel  
        u_id, k_id, w_id, fsq_id, lap_id, m_id = [configuration.sid[key] for key in ['u', 'k', 'w', 'fsq', 'lap', 'm']]
        v_id, sx_id = [configuration.vectors.indices[key] for key in ['v', 'sx']]

        volume_function = numba.njit(configuration.simbox.volume)

        def kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_action_params):
            """     
            """
            steps_between_output, output_array = runtime_action_params # Needs to be compatible with get_params above
            if step%steps_between_output==0:
                save_index = step//steps_between_output
            
                global_id, my_t = cuda.grid(2)
                if global_id < num_part and my_t == 0:
                    cuda.atomic.add(output_array, (save_index, 0), scalars[global_id][u_id])   # Potential energy
                    cuda.atomic.add(output_array, (save_index, 1), scalars[global_id][w_id])   # Virial
                    cuda.atomic.add(output_array, (save_index, 2), scalars[global_id][lap_id]) # Laplace
                    cuda.atomic.add(output_array, (save_index, 3), scalars[global_id][fsq_id]) # F**2
                    cuda.atomic.add(output_array, (save_index, 4), scalars[global_id][k_id])   # Kinetic energy
                
                    # Contribution to total momentum
                    my_m = scalars[global_id][m_id]
                    cuda.atomic.add(output_array, (save_index, 6), my_m*vectors[v_id][global_id][0])
                    cuda.atomic.add(output_array, (save_index, 7), my_m*vectors[v_id][global_id][1])
                    cuda.atomic.add(output_array, (save_index, 8), my_m*vectors[v_id][global_id][2])

                    # XY component of stress
                    cuda.atomic.add(output_array, (save_index, 9), vectors[sx_id][global_id][1] -
                                    my_m * vectors[v_id][global_id][0]*vectors[v_id][global_id][1])

                if global_id == 0 and my_t == 0:
                    output_array[save_index][5] = volume_function(sim_box)

            return
        
        kernel = cuda.jit(device=gridsync)(kernel)

        if gridsync:
            return kernel  # return device function
        else:
            return kernel[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters


def extract_scalars(data, column_list, first_block=0, D=3):
    """ Extracts scalar data from simulation output.

    Parameters
    ----------

    data : dict
        Output from a Simulation object.
    
    column_list : list of str

    first_block : int
        Index of the first timeblock to extract data from.

    D : int
        Dimension of the simulation.

    Returns
    -------

    tuple
        Tuple of 1D numpy arrays containing the extracted scalar data.
    
    
    Example
    -------

    >>> import numpy as np
    >>> import rumdpy as rp
    >>> sim = rp.get_default_sim()  # Replace with your simulation object
    >>> for block in sim.timeblocks(): pass
    >>> U, W = rp.extract_scalars(sim.output, ['U', 'W'], first_block=1)
    """

    # Indices hardcoded for now (see scalar_calculator above)

    column_indices = {'U':0, 'W':1, 'lapU':2, 'Fsq':3, 'K':4, 'Vol':5}
    momentum_id_str = ['Px', 'Py', 'Pz', 'Pw']
    if D > 4:
        raise ValueError("Label for total momentum components not defined for dimensions greater than 4")
    for k in range(D):
        column_indices[momentum_id_str[k]] = 6+k

    column_indices['Sxy'] = 6 + D

    output_list = []
    for column in column_list:
        output_list.append(data['scalars'][first_block:,:,column_indices[column]].flatten())
    return tuple(output_list)
