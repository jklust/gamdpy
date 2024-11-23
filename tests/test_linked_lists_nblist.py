import sys
import numpy as np
import rumdpy as rp
from numba import cuda, config
import pytest

from hypothesis import given, strategies as st, settings, Verbosity, example

def kernel_from_devicefunc(devicefunc):

    @cuda.jit( device=False)
    def kernel(vectors, scalars, ptype, sim_box, nblist, nblist_parameters):
        grid = cuda.cg.this_grid()
        devicefunc(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
    return kernel

def run_nblist(configuration, nblist, cut, compute_plan, compute_flags):
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1

        
        params = nblist.get_params(max_cut=cut, compute_plan=compute_plan)
        kernel = nblist.get_kernel(configuration, compute_plan, compute_flags, force_update=False)
    
        if compute_plan['gridsync']:
            kernel = kernel_from_devicefunc(kernel)
            kernel[num_blocks, (pb, tp)](configuration.d_vectors, 
                                           configuration.d_scalars, 
                                           configuration.d_ptype, 
                                           configuration.simbox.d_data,
                                           nblist.d_nblist, 
                                           params)
        else:
            kernel(0,               configuration.d_vectors, 
                                           configuration.d_scalars, 
                                           configuration.d_ptype, 
                                           configuration.simbox.d_data,
                                           nblist.d_nblist, 
                                           params)
        #print(compute_plan, nblist.d_nbflag.copy_to_host())
        nbflag = nblist.d_nbflag.copy_to_host()
        assert nbflag[0] == 0
        assert nbflag[1] == 0

def nblist_test(nx, ny, nz, rho=0.8442, pb=None, tp=None, skin=None, gridsync=None, UtilizeNIII=None, cut=2.5, verbose=True):
    
    # Generate configuration with a FCC lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=(nx, ny, nz), rho=rho)
    np.random.seed(0)
    configuration['r'] += np.random.uniform(-.3, +.3, configuration['r'].shape)
    configuration['r'] = configuration['r'][np.random.permutation(configuration.N),:]

    # Allow for overwriting of the default compute_plan
    compute_plan = rp.get_default_compute_plan(configuration)
    if pb!=None:
        compute_plan['pb'] = pb
    if tp!=None:
        compute_plan['tp'] = tp
    if skin!=None:
        compute_plan['skin'] = np.float32(skin)
    if gridsync!=None:
        compute_plan['gridsync'] = gridsync
    if UtilizeNIII!=None:
        compute_plan['UtilizeNb'] = UtilizeNIII
    if verbose:
        print('simbox lengths:', configuration.simbox.lengths)
        print('compute_plan: ', compute_plan)

    compute_flags = rp.get_default_compute_flags()

    configuration.copy_to_device()
    nblist = rp.NbListLinkedLists(configuration, [], 300)
    run_nblist(configuration, nblist, cut, compute_plan, compute_flags)
    nblist_linked_list = nblist.d_nblist.copy_to_host()
    
    #configuration['r'][0,2] += 2*cut # Testing the test: This should make test fail!
    configuration.copy_to_device()
    nblist = rp.NbList2(configuration, [], 300)
    run_nblist(configuration, nblist, cut, compute_plan, compute_flags)
    nblist_N_squared = nblist.d_nblist.copy_to_host()
    
    return nblist_linked_list, nblist_N_squared, compute_plan


@pytest.mark.experimental
@settings(deadline=200_000, max_examples = 8)
@given(nx=st.integers(min_value=12, max_value=32), ny=st.integers(min_value=12, max_value=32), nz=st.integers(min_value=12, max_value=32))
def test_nblist(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    nblist_linked_list, nblist_N_squared, compute_plan = nblist_test(nx, ny, nz, cut=2.5, verbose=False)
    total_num_nbs_linked_list = np.sum(nblist_linked_list[:,-1])
    total_num_nbs_N_squared = np.sum(nblist_N_squared[:,-1])
    print(N, nx, ny, nz,
          compute_plan,
          total_num_nbs_linked_list, total_num_nbs_N_squared, total_num_nbs_linked_list/N)
    assert total_num_nbs_linked_list == total_num_nbs_N_squared
    assert np.all(nblist_linked_list[:,-1] == nblist_N_squared[:,-1]) # Num nbs for each particle
    assert np.all(np.sort(nblist_linked_list, axis=1) == np.sort(nblist_N_squared, axis=1)) # Same nbs, order allowed to differ
    return

if __name__ == "__main__":
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    test_nblist()
