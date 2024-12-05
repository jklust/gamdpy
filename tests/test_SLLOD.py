""" Example of a Simulation using rumdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensemble followed by shearing with SLLOD 
and Lees-Edwards boundary conditions

"""



def test_SLLOD(run_NVT=False):
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    import rumdpy as rp

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    temperature_low = 0.700
    gridsync = True

    if run_NVT:
        # Setup configuration: FCC Lattice
        configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973)
        configuration.randomize_velocities(T=2.0)

        # Setup integrator to melt the crystal
        dt = 0.005
        num_blocks = 10
        steps_per_block = 2048
        running_time = dt*num_blocks*steps_per_block

        Ttarget_function = rp.make_function_ramp(value0=2.000, x0=running_time*(1/8), 
                                                value1=temperature_low, x1=running_time*(7/8))
        integrator_NVT = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

        # Set simulation up. Total number of timesteps: num_blocks * steps_per_block
        sim_NVT = rp.Simulation(configuration, pairpot, integrator_NVT,
                                num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                                storage='memory')


        for block in sim_NVT.run_timeblocks():
            print(block)
            print(sim_NVT.status(per_particle=True))

        rp.configuration_to_hdf5(configuration, 'LJ_cooled_0.70.h5')
    else:
        configuration = None
        possible_file_paths = ['Data/LJ_cooled_0.70.h5', 'tests/Data/LJ_cooled_0.70.h5']
        for path in possible_file_paths:
            if Path(path).is_file():
                configuration = rp.configuration_from_hdf5(path)
                break
        if configuration is None:
            raise FileNotFoundError(f'Could not find configuration file in {possible_file_paths}')

    compute_plan = rp.get_default_compute_plan(configuration)
    compute_plan['gridsync'] = gridsync
    print('compute_plan')
    print(compute_plan)
    print("Run SLLOD simulation")

    sc_output = 1

    sr = 0.1
    dt = 0.01

    configuration.simbox = rp.Simbox_LeesEdwards(configuration.D, configuration.simbox.lengths)

    integrator_SLLOD = rp.integrators.SLLOD(shear_rate=sr, dt=dt)

    # Test get_kernel
    integrator_SLLOD.get_kernel(configuration=configuration,
                                compute_plan = rp.get_default_compute_plan(configuration),
                                compute_flags = rp.get_default_compute_flags(),
                                verbose=True)

    # set the kinetic temperature to the exact value associated with the desired
    # temperature since SLLOD uses an isokinetic thermostat
    configuration.set_kinetic_temperature(temperature_low, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

    # Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
    sim_SLLOD = rp.Simulation(configuration, pairpot, integrator_SLLOD,
                            num_timeblocks=3, steps_per_timeblock=128, scalar_output=sc_output,
                            steps_between_momentum_reset=100,
                            storage='memory', compute_flags={'stresses':True}, compute_plan=compute_plan, include_simbox_in_output=True)

    # Run simulation one block at a time
    for block in sim_SLLOD.run_timeblocks():
        print(sim_SLLOD.status(per_particle=True))
        configuration.simbox.copy_to_host()
        box_shift = configuration.simbox.box_shift
        lengths = configuration.simbox.lengths
        print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
    print(sim_SLLOD.summary())

    sxy = rp.extract_scalars(sim_SLLOD.output, ['Sxy'])/configuration.get_volume()

    sxy_mean = np.mean(sxy)
    print(f'{sr:.2g} {sxy_mean:.6f}')
    assert (np.isclose(sxy_mean, 0.876, atol=0.005 ))
    assert(np.isclose(sim_SLLOD.nbflag[2], 49, atol=1))

if __name__ == '__main__':
    test_SLLOD()
