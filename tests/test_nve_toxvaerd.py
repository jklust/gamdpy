
def test_nve_toxvaerd(verbose=False, plot_figures=False):
    """ Compare the Toxvaerd NVE integrator with the standard NVE integrator.
    Assert that the kinetic and configurational temperatures are (almost) the same.
    """
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import rumdpy as rp

    density: float = 0.9
    dt: float = 0.010
    temperature: float = 1.2

    # Setup configuration
    configuration = rp.make_configuration_fcc(nx=7, ny=7, nz=7, rho=density, T=2 * temperature)

    # Setup interactions
    pairfunc = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator
    integrator = rp.integrators.NVE_Toxvaerd(dt=dt)

    # Setup the Simulation
    num_blocks = 32
    steps_per_block = 512
    sim = rp.Simulation(configuration, pairpot, integrator,
                        num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                        conf_output=None, storage='memory', verbose=False)
    
    # Run simulation one block at a time
    for block in sim.timeblocks():
        pass 
    print(sim.summary())

    # Convert scalars to dataframe
    columns = ['U', 'W', 'lapU', 'Fsq', 'K']
    data = np.array(rp.extract_scalars(sim.output, columns, first_block=1))
    df_toxverd = pd.DataFrame(data.T, columns=columns) 

    # Run standard NVE Simulation
    integrator = rp.integrators.NVE(dt=dt)
    sim = rp.Simulation(configuration, pairpot, integrator,
                        num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                        conf_output=None, storage='memory', verbose=False)
    for block in sim.timeblocks():
        pass 
    print(sim.summary())
    data = np.array(rp.extract_scalars(sim.output, columns, first_block=1))
    df_nve = pd.DataFrame(data.T, columns=columns) 

    def compute_T_conf(df):
        N = configuration.N
        D = configuration.D
        df['T_kin'] = 2 * df['K'] / D / (N - 1)
        df['T_conf'] = df['Fsq'] / df['lapU']
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