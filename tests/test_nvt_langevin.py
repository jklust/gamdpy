def test_step_langevin(verbose=True, plot_figures=True) -> None:
    """ Test NVT langevin thermostat
    Test temperature T=1.2 (r_c=2.5) fcc-liquid coexistence state-point in https://doi.org/10.1063/1.4818747 """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from numba.cuda.random import create_xoroshiro128p_states

    import rumdpy as rp

    # State-point
    temperature = 1.2
    density = 1 / 0.9672

    # Expected values
    expected_kinetic_energy = 3 / 2 * temperature
    expected_total_energy = -4.020
    expected_potential_energy = expected_total_energy - expected_kinetic_energy

    # Setup configuration (give temperature kick to particles to get closer to equilibrium)
    conf = rp.make_configuration_fcc(nx=7, ny=7, nz=7, rho=density, T=2*temperature)
    conf.copy_to_device()

    # Setup compute plan
    compute_plan = rp.get_default_compute_plan(conf)
    if verbose:
        print('compute_plan: ', compute_plan)

    # Setup pseudo-random number generator
    number_of_particles = conf.N
    dimension_of_space = conf.D
    random_numbers_per_particle = dimension_of_space
    number_of_random_numbers = number_of_particles * random_numbers_per_particle
    rng_states = create_xoroshiro128p_states(number_of_random_numbers, seed=2023)

    # Setup interactions
    pair_potential = rp.apply_shifted_potential_cutoff(rp.make_LJ_m_n(12, 6))
    params = [[[4.0, -4.0, 2.5], ], ]
    lennard_jones = rp.PairPotential(conf, pair_potential,
                                     params=params, max_num_nbs=1000, compute_plan=compute_plan)
    pairs = lennard_jones.get_interactions(conf, exclusions=None, compute_plan=compute_plan, verbose=verbose)

    # Setup integrator
    dt = 0.005
    alpha = 0.1
    integrator_step = rp.make_step_nvt_langevin(conf, lambda t: np.float32(temperature),
                                                compute_plan=compute_plan, verbose=verbose)
    integrate = rp.make_integrator(conf, integrator_step, pairs['interactions'],
                                   compute_plan=compute_plan, verbose=verbose)
    integrator_params = np.float32(dt), np.float32(alpha), rng_states
    inner_steps = 32
    outer_steps = 512

    # JIT compile
    tic = time.perf_counter()
    integrate(conf.d_vectors, conf.d_scalars, conf.d_ptype, conf.d_r_im,
              conf.simbox.d_data, pairs['interaction_params'], integrator_params, np.float32(0), inner_steps)
    toc = time.perf_counter()
    if verbose:
        print(f"Time spent (JIT): {toc - tic:0.4f} seconds")

    # Run simulation
    scalars = []
    tic = time.perf_counter()
    for i in range(outer_steps):
        integrate(conf.d_vectors, conf.d_scalars, conf.d_ptype, conf.d_r_im, conf.simbox.d_data,
                  pairs['interaction_params'], integrator_params, np.float32(0.0), inner_steps)
        scalars.append(np.sum(conf.d_scalars.copy_to_host(), axis=0))
    toc = time.perf_counter()
    steps_per_second = outer_steps * inner_steps / (toc - tic)
    if verbose:
        print(f"Time spent: {toc - tic:0.4f} seconds")
        print(f"Steps per second: {steps_per_second:0.1f}")

    # Create DataFrame and plot energies
    df = pd.DataFrame(np.array(scalars), columns=conf.sid.keys())
    if plot_figures:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df['u'] / conf.N, label='u')
        plt.subplot(2, 1, 2)
        plt.plot(df['k'] / conf.N, label='k')
    df = df.iloc[len(df) // 2:, :]  # last half
    if plot_figures:
        plt.subplot(2, 1, 1)
        plt.plot(df['u'] / conf.N, label='u (last half)')
        plt.plot([0, 2*len(df)], [expected_potential_energy, expected_potential_energy], 'k--', label='expected')
        plt.ylabel(r'Potential energy, $u$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df['k'] / conf.N, label='k (last half)')
        plt.plot([0, 2*len(df)], [expected_kinetic_energy, expected_kinetic_energy], 'k--', label='expected')
        plt.ylabel(r'Kinetic energy, $k$')
        plt.xlabel('Outer loop step')
        plt.legend()
        plt.show()

    # Compute summary statistics
    summary_statistics = df.describe()
    potential_energy = summary_statistics['u']['mean'] / conf.N  # per particle
    kinetic_energy = summary_statistics['k']['mean'] / conf.N  # per particle
    if verbose:
        print(
            f'Potential energy (per particle): {potential_energy: 8.4f} (expected: {expected_potential_energy: 8.4f})')
        print(f'Kinetic energy (per particle):   {kinetic_energy: 8.4f} (expected: {expected_kinetic_energy: 8.4f})')

    # Assert that the energies are close to the expected values
    assert abs(potential_energy - expected_potential_energy) < 0.05
    assert abs(kinetic_energy - expected_kinetic_energy) < 0.05

    # Ensure that the implementation is fast
    assert steps_per_second > 1000

    return
