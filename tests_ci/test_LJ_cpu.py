import pytest

# check on personalized pytest mark
@pytest.mark.rumdpy_ci
def test_cpu(nconf='1', integrator_type='NVE', potential='KABLJ'):
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
    import rumdpy as rp
    import numpy as np
    import numba
    from numba import cuda
    print(f"Testing configuration={nconf}, integrator_type={integrator_type} and potential={potential}, nunba version: {numba.__version__}")
        
    # Generate configurations with a FCC lattice
    # NOTE: if nx,ny,nz are lower than 4,2,4 fails (in any order)
    # NOTE: some combinations systematically fails as 4,5,4
    configuration = rp.Configuration(D=3)
    if   nconf == '1':
        configuration.make_lattice(rp.unit_cells.FCC, cells=[4, 4, 2], rho=0.8442)
        configuration['m'] = 1.0
        configuration.randomize_velocities(T=1.44)
    elif nconf == '2':
        configuration.make_lattice(rp.unit_cells.FCC, cells=[4, 3, 4], rho=1.2000)
        configuration['m'] = 1.0
        configuration.randomize_velocities(T=0.44)
    elif nconf == '3':
        configuration.make_lattice(rp.unit_cells.FCC, cells=[4, 4, 4], rho=0.8442)
        configuration['m'] = 1.0
        configuration.randomize_velocities(T=2.44)
    else:
        print("wrong input")
        exit()
    isinstance(configuration, rp.Configuration)

    # Make pair potentials
    if   potential == 'LJ':
        pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
        sig, eps, cut = 1.0, 1.0, 2.5
        pairpot = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)
    elif potential == 'KABLJ':
        pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
        sig = [[1.00, 0.80],
               [0.80, 0.88]]
        eps = [[1.00, 1.50],
               [1.50, 0.50]]
        cut = np.array(sig)*2.5
        pairpot = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)
    else:
        print("wrong input")
        exit()
    isinstance(pairpot, rp.PairPotential)

    # Make integrators
    dt = 0.005 # timestep 
    temperature = 0.7 # Not used for NVE
    pressure    = 1.2 # Not used for NV*

    if   integrator_type == 'NVE':
        integrator = rp.integrators.NVE(dt=dt)
        assert isinstance(integrator, rp.integrators.NVE)
    elif integrator_type == 'NVE_Toxvaerd':
        integrator = rp.integrators.NVE_Toxvaerd(dt=dt)
        assert isinstance(integrator, rp.integrators.NVE_Toxvaerd)
    elif integrator_type == 'NVT':
        integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)
        assert isinstance(integrator, rp.integrators.NVT)
    elif integrator_type == 'NPT_Atomic':
        integrator = rp.integrators.NPT_Atomic(temperature=temperature, tau=0.4, pressure=pressure, tau_p=20, dt=dt)
        assert isinstance(integrator, rp.integrators.NPT_Atomic)
    elif integrator_type == 'NVT_Langevin':
        integrator = rp.integrators.NVT_Langevin(temperature=temperature, alpha=0.2, dt=dt, seed=2023)
        assert isinstance(integrator, rp.integrators.NVT_Langevin)
    elif integrator_type == 'NPT_Langevin':
        integrator = rp.integrators.NPT_Langevin(temperature=temperature, pressure=pressure, 
                                                alpha=0.1, alpha_baro=0.0001, mass_baro=0.0001, 
                                                volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2,
                                                dt=dt, seed=2023)
        assert isinstance(integrator, rp.integrators.NPT_Langevin)
    else:
        print("wrong input")
        exit()

    ev = rp.Evaluater(configuration, pairpot)
    sim = rp.Simulation(configuration, pairpot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=64, steps_per_timeblock=1024, storage='memory')
    assert isinstance(sim, rp.Simulation)
    #cuda.simulator.reset()
    #del os.environ["NUMBA_ENABLE_CUDASIM"]
    #del os.environ["NUMBA_DISABLE_JIT"]
    #del os.environ["NUMBA_CUDA_DEBUGINFO"]

if __name__ == '__main__':
    #import sys
    #test_cpu(*sys.argv[1:])
    for configuration in ['1', '2', '3']:
        for integrator in ['NVE', 'NVT', 'NPT_Atomic']:
            for potential in ['LJ', 'KABLJ']:
                test_cpu(nconf=configuration, integrator_type=integrator, potential=potential)
