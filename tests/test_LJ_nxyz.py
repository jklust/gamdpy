import numpy as np
import rumdpy as rp
from numba import cuda, config
import pandas as pd

from hypothesis import given, strategies as st, settings, Verbosity, example

def LJ(nx, ny, nz, rho=0.8442, pb=None, tp=None, skin=None, gridsync=None, UtilizeNIII=None, cut=2.5, integrator='NVE', verbose=True):
    
    # Generate configuration with a FCC lattice
    c1 = rp.make_configuration_fcc(nx=nx,  ny=ny,  nz=nz,  rho=rho, T=1.44) #
    assert c1.N==nx*ny*nz*4, f'Wrong number particles (FCC), {C1.N} <> {nx*ny*nz*4}'
    assert c1.D==3, f'Wrong dimension (FCC), {c1.D} <> {3}'
    c1.copy_to_device()    
    
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
   
    # Make the pair potential.
    pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
    params = [[[4.0, -4.0, 2.5],], ]
    pair_potential = rp.PairPotential(c1, pairpot_func, params=params, max_num_nbs=1000, compute_plan=compute_plan)
    pairs = pair_potential.get_interactions(c1, exclusions=None, compute_plan=compute_plan, verbose=False)
       
    # Setup the integrator
    dt = np.float32(0.005)
    steps = 250
    inner_steps = 40

    if integrator=='NVE':
        step = rp.make_step_nve(c1, compute_plan=compute_plan, verbose=False, )
        integrator_params =  (dt, )
    if integrator=='NVT':
        T = np.float32(0.7)
        tau=0.2
        omega2 = np.float32(4.0*np.pi*np.pi/tau/tau)
        degrees = c1.N*c1.D - c1.D
        thermostat_state = np.zeros(2, dtype=np.float32)
        d_thermostat_state = cuda.to_device(thermostat_state)

        step = rp.make_step_nvt(c1, compute_plan=compute_plan, verbose=False, )
        integrator_params =  (dt, T, omega2, degrees,  d_thermostat_state)
    integrate = rp.make_integrator(c1, step, pairs['interactions'], compute_plan=compute_plan, verbose=False) 
     
    # Run the simulation
    scalars_t = []
    tt = []
    for i in range(steps):
        integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, 
                  pairs['interaction_params'], integrator_params, np.float32(0.0), inner_steps)
        scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
        tt.append(i*inner_steps*dt)            
   
    df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
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

@settings(deadline=200_000, max_examples = 15)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
@example(nx=4,  ny=4,  nz=4)
@example(nx=16, ny=16, nz=32)
def test_nve(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVE', cut=2.5, verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    print(N, '\t', nx, '\t', ny, '\t', nz, '\t', var_e, '\t', Tkin, '\t',Tconf, '\t',R, '\t',Gamma)
    assert var_e < 0.001
    assert 0.68 < Tkin  < 0.71
    assert 0.68 < Tconf < 0.71
    assert 0.89 <   R   < 0.97
    assert 5.2  < Gamma < 6.5

@settings(deadline=200_000, max_examples = 15)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
@example(nx=4,  ny=4,  nz=4)
@example(nx=16, ny=16, nz=32)
def test_nvt(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVT', verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    print(N, '\t', nx, '\t', ny, '\t', nz, '\t', var_e, '\t', Tkin, '\t',Tconf, '\t',R, '\t',Gamma)
    # assert var_e < 0.001
    assert 0.68 < Tkin  < 0.72
    assert 0.68 < Tconf < 0.72
    assert 0.92 <   R   < 1.00
    assert 5.2  < Gamma < 6.8
 
    
if __name__ == "__main__":
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    print('Testing LJ NVE:')
    test_nve()
    print('Testing LJ NVT:')
    test_nvt()

