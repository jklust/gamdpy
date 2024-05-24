import glob
import sys
from rumdpy.integrators import nve, nvt_nh, nvt_langevin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import cuda, config

import rumdpy as rp

def setup_lennard_jones_system(nx, ny, nz, rho=0.8442, cut=2.5, verbose=False):
    """
    Setup and return configuration, potential function, and potential parameters for the LJ benchmark
    """

    # Generate configuration with a FCC lattice
    c1 = rp.make_configuration_fcc(nx=nx,  ny=ny,  nz=nz,  rho=rho, T=1.44) #
    
    # Setup pair potential.
    pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)
    
    return c1, pairpot


def run_benchmark(c1, pairpot, compute_plan, steps, integrator='NVE', verbose=False):
    """
    Run LJ benchmark
    Could be run with other potential and/or parameters, but asserts would need to be updated
    """
    
    # Set up the integrator
    dt = np.float32(0.005)

    #if integrator == 'NVE':
    #    integrate, integrator_params = nve.setup(c1, pairs['interactions'], dt=dt, compute_plan=compute_plan, verbose=False)
    if integrator == 'NVT':
        integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)
    #if integrator=='NVT_Langevin':
    #    integrate, integrator_params = nvt_langevin.setup(c1, pairs['interactions'], T0, alpha=0.1, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

    # Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
    sim = rp.Simulation(c1, pairpot, integrator, 
                    num_blocks=1, steps_per_block=steps, 
                    conf_output=None, scalar_output=None, 
                    storage='memory', verbose=False)

    # Run simulation one block at a time
    for block in sim.blocks():
        pass
    for block in sim.blocks():
        pass

    #print(sim.summary())
    
    #assert 0.55 < Tkin < 0.85, f'{Tkin=}'
    #assert 0.55 < Tconf < 0.85, f'{Tconf=}'
    #if integrator == 'NVE':  # Only expect conservation of energy if we are running NVE
    #    assert -0.01 < de < 0.01
    #assert nbflag[0] == 0
    #assert nbflag[1] == 0

    tps = sim.last_num_blocks*sim.steps_per_block/sim.timing_numba*1000
    time_in_sec = sim.timing_numba/1000
    
    #print(c1.N, '\t', tps, '\t', steps, '\t', time_in_sec, '\t', compute_plan, '\t', Tkin, '\t', Tconf, '\t', de)
    print(c1.N, '\t', tps, '\t', steps, '\t', time_in_sec, '\t', compute_plan)

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
        c1, LJ_func = setup_lennard_jones_system(*nxyz, cut=2.5, verbose=False)
        time_in_sec = 0
        while time_in_sec < 1.0:  # At least 1s to get reliable timing
            steps = int(magic_number / c1.N)
            compute_plan = rp.get_default_compute_plan(c1)
            # compute_plan['tp'] = 1
            tps, time_in_sec = run_benchmark(c1, LJ_func, compute_plan, steps, integrator=integrator, verbose=False)
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
    plt.legend(loc='lower left', fontsize=6)
    plt.ylim(1, 1e6)
    plt.xlabel('N')
    plt.ylabel('TPS')
    plt.savefig('Data/benhcmarks.pdf')
    plt.show()

    # Save this run to csv file
    df.to_csv('Data/benchmark_LJ_Last_run.csv', index=False)


if __name__ == "__main__":
    #if len(sys.argv)==1 or 'NVE' in sys.argv:
    #    main(integrator='NVE')
    if 'NVT' in sys.argv:
        main(integrator='NVT')
    #if 'NVT_Langevin' in sys.argv:
    #    main(integrator='NVT_Langevin')
        

