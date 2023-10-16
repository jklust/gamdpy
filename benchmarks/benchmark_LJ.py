import numpy as np
import rumdpy as rp
from numba import cuda, config
import pandas as pd
import matplotlib.pyplot as plt


def LJ(nx, ny, nz, rho=0.8442, cut=2.5, verbose=True):
    # Setup configuration and potential function for the LJ benchmark
    
    # Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density
    positions, simbox_data = rp.generate_fcc_positions(nx=nx, ny=ny, nz=nz, rho=rho)
    N, D = positions.shape
    assert N==nx*ny*nz*4, f'Wrong number particles (FCC), {N} <> {nx*ny*nz*4}'
    assert D==3, f'Wrong dimension (FCC), {D} <> {3}'

    # make 'random' velocities reproducible
    np.random.seed(31415)
    
    ### Make configuration. Could be read from file or generated from single convenience function, but this shows flexibility
    c1 = rp.Configuration(N, D, simbox_data)
    c1['r'] = positions
    c1['v'] = rp.generate_random_velocities(N, D, T=1.44)
    c1['m'] =  np.ones(N, dtype=np.float32)     # Set masses
    c1.ptype = np.zeros(N, dtype=np.int32)      # Set types
 
    LJ_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)

    params = np.zeros((1,1), dtype="f,f,f")
    params[0][0] = (4., -4., 2.5)
    if verbose:
        print('Pairpotential paramaters:\n', params)
        print('simbox_data:', simbox_data)
        
    return c1, LJ_func, params

    
def run(c1, pairpot_func, params, compute_plan, steps, integrator='NVE', verbose=True):
    # run benchmark 

    if verbose:
        print('compute_plan: ', compute_plan)
   
    # Make the pair potential. NOTE: params is a 2 dimensional numpy array of tuples
    pair_potential = rp.PairPotential(c1, pairpot_func, params=params, max_num_nbs=1000, compute_plan=compute_plan)
    num_cscalars = 3

    c1.copy_to_device()                
    pair_potential.copy_to_device()
    
    interactions = rp.make_interactions(c1, pair_potential=pair_potential, num_cscalars=num_cscalars, 
                                        compute_plan=compute_plan, verbose=verbose,)
    interaction_params = (pair_potential.d_params, compute_plan['skin'], 
                          pair_potential.nblist.d_nblist,  pair_potential.nblist.d_nbflag)
   
    # Setup the integrator
    dt = np.float32(0.005)

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
    integrate = rp.make_integrator(c1, step, interactions, compute_plan=compute_plan, verbose=False) 
     
    # Run the simulation
    
    
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, 10)
    
    scalars_t = []
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
    start = cuda.event()
    end = cuda.event()

    start.record()
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, steps)        
    end.record()
    end.synchronize()
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))

    nbflag = pair_potential.nblist.d_nbflag.copy_to_host()    
    time_in_sec = np.float32(cuda.event_elapsed_time(start, end)/1000)
    tps = np.float32(steps/time_in_sec)
    
    if verbose:
        print('\tsteps :', steps)
        print('\tnbflag : ', nbflag)
        print('\ttime :', time_in_sec, 's')
        print('\tTPS : ', tps )
    
    df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
    if compute_plan['UtilizeNIII']:
        df['u'] *= 2
        df['w'] *= 2
        df['lap'] *= 2
    df['w'] *= 1/c1.D/2
    df['e'] = df['u'] + df['k'] # Total energy
    df['Tkin'] =2*df['k']/c1.D/(c1.N-1)
    df['Tconf'] = df['fsq']/df['lap']
    
    Tkin = df['Tkin'][1]
    Tconf = df['Tconf'][1]
    de = np.float32((df['e'][1] - df['e'][0])/c1.N)
    
    print(c1.N, '\t', tps, '\t',  steps, '\t', time_in_sec, '\t', compute_plan, '\t', Tkin, '\t', Tconf, '\t', de)
    
    assert  0.65 < Tkin    < 0.75
    assert  0.6  < Tconf   < 0.8
    assert -0.01 < de < 0.01
    assert nbflag[0] == 0
    assert nbflag[1] == 0

    return tps, time_in_sec


if __name__ == "__main__":
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    print('Benchmarking LJ NVE:')
    nxyzs = ((4,4,8), (4,8,8), (8,8,8), (8,8,16), (8,16,16), (16,16,16), (16,16,32), (16,32,32), (32,32,32))
    Ns = []
    tpss = []
    magic_number = 1e7
    for nxyz in nxyzs:
        c1, LJ_func, params = LJ(*nxyz, verbose=False)
        time_in_sec = 0
        while time_in_sec < 1.0:             # At least 1s to get reliable timing
            steps = int(magic_number/c1.N)
            compute_plan = rp.get_default_compute_plan(c1)
            tps, time_in_sec = run(c1, LJ_func, params, compute_plan, steps, integrator='NVE', verbose=False)
            magic_number *= 2.0/time_in_sec   # Aim for 2 seconds (Assuming O(N) scaling)
        Ns.append(c1.N)
        tpss.append(tps)
    
    N = np.array(Ns)
    tps = np.array(tpss)
    
    plt.figure()
    plt.title('LJ benchmark')
    plt.loglog(N, tps, 'o-', label='This run')
    plt.loglog(N, 200*1e6/N, '--', label='Perfect scaling (MATS=200)')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('TPS')
    plt.show()
    