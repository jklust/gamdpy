""" Check CUDA availability, versions, and test if gridsync is supported. """




def check_cuda(verbose=True):
    """ Check CUDA availability, versions, and test if gridsync is supported. Returns True if gridsync is supported."""
    import numba
    from numba import cuda

    if verbose:
        print('  ..:: CUDA information ::..')
        print(f'{numba.__version__ = }')
        print(f'{numba.cuda.is_available() = }')
        print(f'{numba.cuda.is_supported_version() = }')
        print(f'{cuda.runtime.get_version() = }')


def gridsync_example():
    """ Example from https://numba.readthedocs.io/en/stable/cuda/cooperative_groups.html """
    from numba import cuda, int32, config
    import numpy as np
 
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    config.CUDA_WARN_ON_IMPLICIT_COPY = False
    
    sig = (int32[:, ::1],)

    @cuda.jit(sig)
    def sequential_rows(M):
        col = cuda.grid(1)
        g = cuda.cg.this_grid()

        rows = M.shape[0]
        cols = M.shape[1]

        for row in range(1, rows):
            opposite = cols - col - 1
            M[row, col] = M[row - 1, opposite] + 1
            g.sync()

    A = np.zeros((1024, 1024), dtype=np.int32)
    blockdim = 32
    griddim = A.shape[1] // blockdim
    sequential_rows[griddim, blockdim](A)


    overload = sequential_rows.overloads[(int32[:, ::1],)]
    max_blocks = overload.max_cooperative_grid_blocks(blockdim)
    return max_blocks

def check_gridsync(verbose=True):
    """ Check if gridsync is supported. Returns True if gridsync is supported."""
    import numba
    from numba import cuda
    # See if gridsync is working
    max_blocks = None
    try:
        max_blocks = gridsync_example()
    except numba.cuda.cudadrv.driver.LinkerError as e:
        if verbose:
            print('Warning: gridsync is not supported. Try this hack:')
            print('Find where libcudadevrt.a is located, and write something like this')
            print('    ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .')
            print('in the directory where you run the code.')
        return False

    if verbose:
        print('  ..:: Gridsync check ::..')
        print('Confirmed that gridsync is supported by executing test code.')
        print(f'{max_blocks = }')
    return True

def check_gpu(device_id=None):
    """ Print some information about the GPU. """
    from numba import cuda

    # Get the device
    if device_id==None:
        device = cuda.get_current_device()
    else:
        device = cuda.select_device(device_id)
        
    from rumdpy.cc_cores_per_SM_dict import cc_cores_per_SM_dict 

    from cc_cores_per_SM_dict import cc_cores_per_SM_dict 
    if device.compute_capability in cc_cores_per_SM_dict:
        cc_cores_per_SM = cc_cores_per_SM_dict[device.compute_capability]
    else:
        print('WARNING: Could not find cc_cores_per_SM for this compute_capability. Guessing: 128')
        cc_cores_per_SM=128
    
    # Print relevant attributes
    print('  ..:: GPU information ::..')
    print("Device Name:", device.name)
    print("Compute Capability:", device.compute_capability)
    print("Number of Streaming Multiprocessors:", device.MULTIPROCESSOR_COUNT)
    print("Total number of cores:", cc_cores_per_SM*device.MULTIPROCESSOR_COUNT)
    print("Max Threads Per Block:", device.MAX_THREADS_PER_BLOCK)
    print("Max Block Dimensions (x, y, z):",
          device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y, device.MAX_BLOCK_DIM_Z)
    print("Max Grid Dimensions (x, y, z):",
          device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z)
    print("Max Shared Memory Per Block:", device.MAX_SHARED_MEMORY_PER_BLOCK)
    print("Total Constant Memory:", device.TOTAL_CONSTANT_MEMORY)
    print("Warp Size:", device.WARP_SIZE)
    print("L2 cache size:", device.L2_CACHE_SIZE)
    print("Max registers per block:", device.MAX_REGISTERS_PER_BLOCK)
    print("Single to double performance ratio:", device.SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
 

if __name__ == '__main__':
    check_cuda()
    check_gridsync()
    check_gpu()
