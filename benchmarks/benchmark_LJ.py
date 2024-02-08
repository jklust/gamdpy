import glob
import sys
from rumdpy.integrators import nve, nvt_nh, nvt_langevin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import cuda, config

import rumdpy as rp

def setup_lennard_jones_system(nx, ny, nz, rho=0.8442, cut=2.5, verbose=True):
    """
    Setup and return configuration, potential function, and potential parameters for the LJ benchmark
    """

    # Generate configuration with a FCC lattice
    c1 = rp.make_configuration_fcc(nx=nx,  ny=ny,  nz=nz,  rho=rho, T=1.44) #
    c1.copy_to_device() 

    pairpot_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
    params = [[[4.0, -4.0, 2.5],], ]
    
    return c1, pairpot_func, params


def run_benchmark(c1, pairpot_func, params, compute_plan, steps, integrator='NVE', verbose=True):
    """
    Run LJ benchmark
    Could be run with other potential and/or parameters, but asserts would need to be updated
    """
    if verbose:
        print('compute_plan: ', compute_plan)

    c1.copy_to_device()
    
    # Make the pair potential. 
    pair_potential = rp.PairPotential(c1, pairpot_func, params=params, exclusions=None, max_num_nbs=1000, compute_plan=compute_plan)
    pairs = pair_potential.get_interactions(c1, exclusions=None, compute_plan=compute_plan, verbose=False)
    
    # Set up the integrator
    dt = np.float32(0.005)

    T0 = rp.make_function_constant(value=0.7) # Not used for NVE
    if integrator == 'NVE':
        integrate, integrator_params = nve.setup(c1, pairs['interactions'], dt=dt, compute_plan=compute_plan, verbose=False)
    if integrator == 'NVT':
        integrate, integrator_params = nvt_nh.setup(c1, pairs['interactions'], T0, tau=0.2, dt=dt, compute_plan=compute_plan, verbose=False)
    if integrator=='NVT_Langevin':
        integrate, integrator_params = nvt_langevin.setup(c1, pairs['interactions'], T0, alpha=0.1, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

    # Run the Simulation
    zero = np.float32(0.0)
    # Warmup
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, pairs['interaction_params'],
              integrator_params, zero, 10)

    scalars_t = [np.sum(c1.d_scalars.copy_to_host(), axis=0)]
    start = cuda.event()
    end = cuda.event()

    start.record()
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, pairs['interaction_params'],
              integrator_params, zero, steps)
    end.record()
    end.synchronize()
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))

    nbflag = pair_potential.nblist.d_nbflag.copy_to_host()
    time_in_sec = np.float32(cuda.event_elapsed_time(start, end) / 1000)
    tps = np.float32(steps / time_in_sec)

    if verbose:
        print('\tsteps :', steps)
        print('\tnbflag : ', nbflag)
        print('\ttime :', time_in_sec, 's')
        print('\tTPS : ', tps)

    df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
    df['e'] = df['u'] + df['k']  # Total energy
    df['Tkin'] = 2 * df['k'] / c1.D / (c1.N - 1)
    df['Tconf'] = df['fsq'] / df['lap']

    Tkin = df['Tkin'][1]
    Tconf = df['Tconf'][1]
    de = np.float32((df['e'][1] - df['e'][0]) / c1.N)

    print(c1.N, '\t', tps, '\t', steps, '\t', time_in_sec, '\t', compute_plan, '\t', Tkin, '\t', Tconf, '\t', de)

    assert 0.6 < Tkin < 0.8
    assert 0.6 < Tconf < 0.8
    if integrator == 'NVE':  # Only expect conservation of energy if we are running NVE
        assert -0.01 < de < 0.01
    assert nbflag[0] == 0
    assert nbflag[1] == 0

    return tps, time_in_sec


def main(integrator):
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    print(f'Benchmarking LJ with {integrator} integrator:')
    nxyzs = (
        (4, 4, 8), (6, 6, 6), (4, 8, 8), (8, 8, 8), (8, 8, 16), (8, 16, 16), (16, 16, 16), (16, 16, 32), (16, 32, 32),
        (32, 32, 32))
    Ns = []
    tpss = []
    magic_number = 1e7
    for nxyz in nxyzs:
        c1, LJ_func, params = setup_lennard_jones_system(*nxyz, cut=2.5, verbose=False)
        time_in_sec = 0
        while time_in_sec < 1.0:  # At least 1s to get reliable timing
            steps = int(magic_number / c1.N)
            compute_plan = rp.get_default_compute_plan(c1)
            # compute_plan['tp'] = 1
            tps, time_in_sec = run_benchmark(c1, LJ_func, params, compute_plan, steps, integrator=integrator, verbose=False)
            magic_number *= 2.0 / time_in_sec  # Aim for 2 seconds (Assuming O(N) scaling)
        Ns.append(c1.N)
        tpss.append(tps)

    df = pd.DataFrame({'N': Ns, 'TPS': tpss})
    files_with_benchmark_data = sorted(glob.glob('Data/benchmark_LJ_*.csv'))

    plt.figure()
    plt.title('LJ benchmark, NVE, rho=0.8442')
    plt.loglog(df['N'], df['TPS'], 'o-', label='This run')
    for file in files_with_benchmark_data:
        bdf = pd.read_csv(file)
        label = " ".join(file.split("/")[-1].split('.')[0].split("_")[2::])
        plt.loglog(bdf['N'], bdf['TPS'], '.-', label=label)
    plt.loglog(df['N'], 200 * 1e6 / df['N'], '--', label='Perfect scaling (MATS=200)')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('TPS')
    plt.savefig('Data/benhcmarks.pdf')
    plt.show()

    # Save this run to csv file
    df.to_csv('Data/benchmark_LJ_Last_run.csv', index=False)


if __name__ == "__main__":
    if len(sys.argv)==1 or 'NVE' in sys.argv:
        main(integrator='NVE')
    if 'NVT' in sys.argv:
        main(integrator='NVT')
    if 'NVT_Langevin' in sys.argv:
        main(integrator='NVT_Langevin')
        

