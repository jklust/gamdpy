
def test_nve_toxvaerd(verbose=False, plot_figures=False):
    """ Compare the Toxvaerd NVE integrator with the standard NVE integrator.
    Assert that the kinetic and configurational temperatures are (almost) the same.
    """
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import rumdpy as rp
    from rumdpy.integrators import nve, nve_toxvaerd

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

    # Run equilibration Simulation
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

    # Check that the energy is conserved
    df_toxverd = pd.DataFrame(np.array(scalars), columns=conf.sid.keys())


    # Run standard NVE Simulation
    integrate, integrator_params = nve.setup(conf, pairs['interactions'], dt=dt, compute_plan=compute_plan, verbose=verbose)
    scalars = []
    for i in range(outer_steps):
        integrate(conf.d_vectors, conf.d_scalars, conf.d_ptype, conf.d_r_im, conf.simbox.d_data,
                  pairs['interaction_params'], integrator_params, np.float32(0.0), inner_steps)
        scalars.append(np.sum(conf.d_scalars.copy_to_host(), axis=0))
    df_nve = pd.DataFrame(np.array(scalars), columns=conf.sid.keys())


    def compute_T_conf(df):
        N = conf.N
        D = conf.D
        df['T_kin'] = 2 * df['k'] / D / (N - 1)
        df['T_conf'] = df['fsq'] / df['lap']
        return df

    df_toxverd = compute_T_conf(df_toxverd)
    df_nve = compute_T_conf(df_nve)

    # Check that the temperatures are the same
    T_kin_toxvaerd = df_toxverd['T_kin'].mean()
    T_kin_nve = df_nve['T_kin'].mean()
    T_conf_toxvaerd = df_toxverd['T_conf'].mean()
    T_conf_nve = df_nve['T_conf'].mean()
    if verbose:
        print(f'T_kin (Toxvaerd):  {T_kin_toxvaerd: 8.4f}')
        print(f'T_kin (NVE):       {T_kin_nve: 8.4f}')
        print(f'T_conf (Toxvaerd): {T_conf_toxvaerd: 8.4f}')
        print(f'T_conf (NVE):      {T_conf_nve: 8.4f}')

    # Plot temperatures
    if plot_figures:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df_toxverd['T_conf'], 'r-', label='Toxvaerd (T_conf)')
        plt.plot(df_toxverd['T_kin'], 'b-', label='Toxvaerd (T_kin)')
        plt.plot([0, len(df_toxverd)], [T_conf_toxvaerd, T_conf_toxvaerd], 'k-', lw=4, label='T_conf (avg)')
        plt.plot([0, len(df_toxverd)], [T_kin_toxvaerd, T_kin_toxvaerd], 'y--', lw=4, label='T_kin (avg)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df_nve['T_conf'], 'r-', label='NVE (T_conf)')
        plt.plot(df_nve['T_kin'], 'b-', label='NVE (T_kin)')
        plt.plot([0, len(df_nve)], [T_conf_nve, T_conf_nve], 'k-', lw=4, label='T_conf (avg)')
        plt.plot([0, len(df_nve)], [T_kin_nve, T_kin_nve], 'y--', lw=4, label='T_kin (avg)')
        plt.legend()
        plt.show()

    assert np.isclose(T_kin_toxvaerd, T_conf_toxvaerd, atol=0.01)  # T_kin and T_conf should be the same


if __name__ == '__main__':
    test_nve_toxvaerd(verbose=True, plot_figures=True)