""" Check CUDA availability, versions, and test if gridsync is supported. """


def gridsync_example():
    """ Example from https://numba.readthedocs.io/en/stable/cuda/cooperative_groups.html """
    from numba import cuda, int32

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


def check_gridsync(verbose=True):
    """ Check CUDA availability, versions, and test if gridsync is supported. Returns True if gridsync is supported.

    If gridsync is not supported, try this hack:
        ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .
    in the directory where you run the code.
    """
    import numba
    from numba import cuda

    if verbose:
        print(f'{numba.__version__ = }')
        print(f'{numba.cuda.is_available() = }')
        print(f'{numba.cuda.is_supported_version() = }')
        print(f'{cuda.runtime.get_version() = }')

    # See if gridsync is working
    try:
        gridsync_example()
    except numba.cuda.cudadrv.driver.LinkerError as e:
        if verbose:
            print('Warning: gridsync is not supported. Try this hack:')
            print('Find where libcudadevrt.a is located, and write something like this')
            print('    ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .')
            print('in the directory where you run the code.')
        return False

    return True

if __name__ == '__main__':
    check_gridsync()
