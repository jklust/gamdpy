def test_nve_toxvaerd(verbose = True):
    import numpy as np
    from rumdpy.integrators import nve_toxvaerd
    import rumdpy as rp
    import time

    density: float = 0.9
    dt: float = 0.010
    temperature: float = 1.2

    # Setup configuration
    conf = rp.make_configuration_fcc(nx=7, ny=7, nz=7, rho=density, T=2 * temperature)
    conf.copy_to_device()

    # Setup compute plan
    compute_plan = rp.get_default_compute_plan(conf)
    if verbose:
        print('compute_plan: ', compute_plan)

    # Setup interactions
    pair_potential = rp.apply_shifted_potential_cutoff(rp.make_LJ_m_n(12, 6))
    params = [[[4.0, -4.0, 2.5], ], ]
    lennard_jones = rp.PairPotential(conf, pair_potential,
                                     params=params, max_num_nbs=1000, compute_plan=compute_plan)
    pairs = lennard_jones.get_interactions(conf, exclusions=None, compute_plan=compute_plan, verbose=verbose)

    integrate, integrator_params = nve_toxvaerd.setup(conf, pairs['interactions'], dt=dt,
                                                                     compute_plan=compute_plan, verbose=verbose)

    inner_steps: int = 32
    outer_steps: int = 512

    # JIT compile
    tic: float = time.perf_counter()
    integrate(conf.d_vectors, conf.d_scalars, conf.d_ptype, conf.d_r_im,
              conf.simbox.d_data, pairs['interaction_params'], integrator_params, np.float32(0), inner_steps)
    toc: float = time.perf_counter()
    if verbose:
        print(f"Time spent (JIT): {toc - tic:0.4f} seconds")

    # Run equilibration simulation
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

