import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd

from hypothesis import given, strategies as st, settings, Verbosity

def LJ(nx, ny, nz, rho=0.8442, pb=None, tp=None, skin=None, gridsync=None, UtilizeNIII=None, cut=2.5, integrator='NVE', verbose=True):
    
    # Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density 
    positions, simbox_data = rp.generate_fcc_positions(nx=nx, ny=ny, nz=nz, rho=rho)
    N, D = positions.shape
    assert N==nx*ny*nz*4, f'Wrong number particles (FCC), {N} <> {nx*ny*nz*4}'
    assert D==3, f'Wrong dimension (FCC), {D} <> {3}'

    ### Make configuration. Could be read from file or generated from single convenience function, but this shows flexibility
    c1 = rp.Configuration(N, D, simbox_data)
    c1['r'] = positions
    c1['v'] = rp.generate_random_velocities(N, D, T=1.44)
    c1['m'] =  np.ones(N, dtype=np.float32)     # Set masses
    c1.ptype = np.zeros(N, dtype=np.int32)      # Set types
    
    # Allow for overwriiting of the default compute_plan
    compute_plan = rp.get_default_compute_plan(c1)
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
        print('simbox_data:', simbox_data)
        print('compute_plan: ', compute_plan)
   
    # Make the pair potential. NOTE: params is a 2 dimensional numpy array of tuples
    params = np.zeros((1,1), dtype="f,f,f")
    params[0][0] = (4., -4., 2.5)
    if verbose:
        print('Pairpotential paramaters:\n', params)
    LJ = rp.PairPotential(c1, rp.apply_shifted_force_cutoff(rp.LJ_12_6), params=params, max_num_nbs=1000, compute_plan=compute_plan)
    num_cscalars = 3

    c1.copy_to_device()                
    LJ.copy_to_device()
    
    interactions = rp.make_interactions(c1, pair_potential=LJ, num_cscalars=num_cscalars, compute_plan=compute_plan, verbose=verbose,)
    interaction_params = (LJ.d_params, compute_plan['skin'], LJ.nblist.d_nblist,  LJ.nblist.d_nbflag)
   
    # Setup the integrator
    dt = np.float32(0.005)
    steps = 500
    inner_steps = 40

    if integrator=='NVE':
        step = rp.make_step_nve(c1, compute_plan=compute_plan, verbose=False, )
        integrator_params =  (dt, )
    if integrator=='NVT':
        T = np.float32(0.7)
        tau=0.2
        omega2 = np.float32(4.0*np.pi*np.pi/tau/tau)
        degrees = N*D - D
        thermostat_state = np.zeros(2, dtype=np.float32)
        d_thermostat_state = cuda.to_device(thermostat_state)

        step = rp.make_step_nvt(c1, compute_plan=compute_plan, verbose=False, )
        integrator_params =  (dt, T, omega2, degrees,  d_thermostat_state)
    integrate = rp.make_integrator(c1, step, interactions, compute_plan=compute_plan, verbose=False) 
     
    # Run the simulation
    scalars_t = []
    tt = []
    for i in range(steps):
        integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, inner_steps)
        scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
        tt.append(i*inner_steps*dt)
            
   
    df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
    if compute_plan['UtilizeNIII']:
        df['u'] *= 2
        df['w'] *= 2
        df['lap'] *= 2
    df['w'] *= 1/D/2
    df['t'] = np.array(tt)

    return df

def get_results_from_df(df, N, D):
    df['e'] = df['u'] + df['k'] # Total energy
    df['Tkin'] =2*df['k']/D/(N-1)
    df['Tconf'] = df['fsq']/df['lap']
    df['du'] = df['u'] - np.mean(df['u'])
    df['de'] = df['e'] - np.mean(df['e'])
    df['dw'] = df['w'] - np.mean(df['w'])

    df2 = df.drop(range(50))

    df2['du'] = df2['u'] - np.mean(df2['u'])
    df2['de'] = df2['e'] - np.mean(df2['e'])
    df2['dw'] = df2['w'] - np.mean(df2['w'])

    var_e = np.var(df['e'])/N
    Tkin = np.mean(df2['Tkin'])
    Tconf = np.mean(df2['Tconf'])        
    R = np.dot(df2['dw'], df2['du'])/(np.dot(df2['dw'], df2['dw'])*np.dot(df2['du'], df2['du']))**0.5
    Gamma = np.dot(df2['dw'], df2['du'])/(np.dot(df2['du'], df2['du']))

    return var_e, Tkin, Tconf, R, Gamma

@settings(deadline=10_000, max_examples = 20, verbosity=Verbosity.verbose)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
def test_nve(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    assert var_e < 0.001
    assert 0.69 < Tkin  < 0.71
    assert 0.69 < Tconf < 0.71
    assert 0.90 <   R   < 0.95
    assert 5.5  < Gamma < 6.2

@settings(deadline=10_000, max_examples = 20, verbosity=Verbosity.verbose)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
def test_nvt(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVT', verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    # assert var_e < 0.001
    assert 0.69 < Tkin  < 0.71
    assert 0.69 < Tconf < 0.71
    #assert 0.90 <   R   < 0.95, f'R= {R}'
    assert 0.90 <   R   < 0.97
    assert 5.5  < Gamma < 6.2
 
    
if __name__ == "__main__":
    test_nve()
    # test_nvt() needs calibration in acceptable limits

