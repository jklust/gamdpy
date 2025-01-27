""" 
Benchmark rumdpy using the lammps LJ benchmark
Command line options:

- Nsquared : Use O(N^2) for neighbor-list update (Default)
- LinkedLists : Use O(N) linked lists for neighbor-list update

- NVE : Use NVE integrator (default)
- NVT : Use NVT integrator
- NVT_Langevin : Use NVT_Langevin integrator

"""

import glob
import sys
# from rumdpy.integrators import nve, nvt_nh, nvt_langevin  # OLD CODE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba
from numba import cuda, config

import rumdpy as rp


def setup_lennard_jones_system(nx, ny, nz, rho=0.8442, cut=2.5, verbose=False):
    """
    Setup and return configuration, potential function, and potential parameters for the LJ benchmark
    """

    # Generate configuration with a FCC lattice
    # Setup configuration: FCC Lattice
    c1 = rp.Configuration(D=3)
    c1.make_lattice(rp.unit_cells.FCC, cells=[nx, ny, nz], rho=rho)
    c1['m'] = 1.0
    c1.randomize_velocities(temperature=1.44)
    #  c1 = rp.make_configuration_fcc(nx=nx,  ny=ny,  nz=nz,  rho=rho, T=1.44)

    # Setup pair potential.
    #pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=500)

    return c1, pair_pot


def run_benchmark(c1, pair_pot, compute_plan, steps, integrator='NVE', autotune=False, verbose=False):
    """
    Run LJ benchmark
    Could be run with other potential and/or parameters, but asserts would need to be updated
    """

    # Set up the integrator
    dt = 0.005

    if integrator == 'NVE':
        integrator = rp.integrators.NVE(dt=dt)
    if integrator == 'NVT':
        integrator = rp.integrators.NVT(temperature=0.70, tau=0.2, dt=dt)
    if integrator == 'NVT_Langevin':
        integrator = rp.integrators.NVT_Langevin(temperature=0.70, alpha=0.2, dt=dt, seed=213)
    
    # Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
    sim = rp.Simulation(c1, pair_pot, integrator, [rp.MomentumReset(200), ],
                        num_timeblocks=1, steps_per_timeblock=steps,
                        compute_plan=compute_plan, storage='memory', verbose=False)
    
    

    # Run simulation one block at a time
    for block in sim.run_timeblocks():
        pass
    
    if autotune:
        sim.autotune_bruteforce(verbose=False)

    nbflag0 = pair_pot.nblist.d_nbflag.copy_to_host()
    for block in sim.run_timeblocks():
        pass

    #print(sim.summary())

    #assert 0.55 < Tkin < 0.85, f'{Tkin=}'
    #assert 0.55 < Tconf < 0.85, f'{Tconf=}'
    #if integrator == 'NVE':  # Only expect conservation of energy if we are running NVE
    #    assert -0.01 < de < 0.01
    #assert nbflag[0] == 0
    #assert nbflag[1] == 0

    tps = sim.last_num_blocks * sim.steps_per_block / np.sum(sim.timing_numba_blocks) * 1000
    time_in_sec = sim.timing_numba / 1000

    nbflag = pair_pot.nblist.d_nbflag.copy_to_host()
    nbupdates = nbflag[2] - nbflag0[2]
    print(f"{c1.N:7} {tps:.2e} {steps:.1e} {time_in_sec:.1e}  {nbupdates:6} {steps/nbupdates:.1f} {sim.compute_plan}")
    assert nbflag[0] == 0
    assert nbflag[1] == 0
    return tps, time_in_sec, steps


def main(integrator, nblist, autotune):
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    print(f'Benchmarking LJ with {integrator} integrator:')
    nxyzs = (
        (8, 8, 8), (8, 8, 16), (8, 16, 16), (16, 16, 16), (16, 16, 32), (16, 32, 32), (32, 32, 32) )
    if nblist == 'Nsquared':
        nxyzs = ((4, 4, 8), (4, 8, 8),) + nxyzs
    if nblist == 'LinkedLists':
        nxyzs += (32, 32, 64), (32, 64, 64), (64, 64, 64)
    if nblist == 'default':
        nxyzs = ((4, 4, 8), (4, 8, 8),) + nxyzs
        nxyzs += (32, 32, 64), (32, 64, 64), (64, 64, 64)
    Ns = []
    tpss = []
    tpss_at = []
    magic_number = 1e7
    print('    N     TPS     Steps   Time     NbUpd Steps/NbUpd')
    for nxyz in nxyzs:
        c1, LJ_func = setup_lennard_jones_system(*nxyz, cut=2.5, verbose=False)
        time_in_sec = 0
        while time_in_sec < 0.5:  # At least 1s to get reliable timing
            steps = int(magic_number / c1.N)
            compute_plan = rp.get_default_compute_plan(c1)
            #compute_plan['tp'] = 1
            #compute_plan['tp'] = int(compute_plan['tp']*1.5)
            if nblist=='LinkedLists':
                if c1.N > 2000:
                    compute_plan['nblist'] = 'linked lists'
                    compute_plan['skin'] = 0.3
                    #compute_plan['pb'] = 128
                    #if c1.N < 50000:
                    #    compute_plan['gridsync'] = True
            tps, time_in_sec, steps = run_benchmark(c1, LJ_func, compute_plan, steps, integrator=integrator, verbose=False)
            if autotune:
                tps_at, time_in_sec_at, steps_at = run_benchmark(c1, LJ_func, compute_plan, steps, integrator=integrator, autotune=autotune, verbose=False)
            magic_number *= 1.0 / time_in_sec  # Aim for 2 seconds (Assuming O(N) scaling)
        Ns.append(c1.N)
        tpss.append(tps)
        if autotune:
            tpss_at.append(tps_at)
    
    # Save this run to csv file
    if autotune:  
        df = pd.DataFrame({'N': Ns, 'TPS': tpss, 'TPS_AT':tps_at})
    else:
        df = pd.DataFrame({'N': Ns, 'TPS': tpss})
 
    df.to_csv('Data/benchmark_LJ_Last_run.csv', index=False)
    #files_with_benchmark_data = sorted(glob.glob('Data/benchmark_LJ_*.csv'))

    #plt.figure()
    #plt.title('LJ benchmark, NVE, rho=0.8442')
    #plt.loglog(df['N'], df['TPS'], 'o-', label='This run')
    #for file in files_with_benchmark_data:
    #    bdf = pd.read_csv(file)
    #    label = " ".join(file.split("/")[-1].split('.')[0].split("_")[2::])
    #    plt.loglog(bdf['N'], bdf['TPS'], '.-', label=label)
    #plt.loglog(df['N'], 200 * 1e6 / df['N'], '--', label='Perfect scaling (MATS=200)')
    #plt.legend(loc='lower left', fontsize=6)
    #plt.ylim(1, 1e6)
    #plt.xlabel('N')
    #plt.ylabel('TPS')
    #plt.savefig('Data/benhcmarks.pdf')
    #plt.show()

if __name__ == "__main__":
    integrator = 'NVE'
    nblist = 'Nsquared'
    if 'NVT' in sys.argv:
        integrator = 'NVT'
    if 'NVT_Langevin' in sys.argv:
        integrator = 'NVT_Langevin'

    nblist = 'default'    
    if 'LinkedLists' in sys.argv:
        nblist = 'LinkedLists'
    if 'NSquared' in sys.argv:
        nblist = 'NSquared'

    autotune = 'autotune' in sys.argv

    main(integrator=integrator, nblist=nblist, autotune=autotune)
