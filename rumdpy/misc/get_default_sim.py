"Return a sim object of the single component LJ crystal in the NVT ensemble."

def get_default_sim():
    """ Return a sim object of the single component LJ crystal in the NVT ensemble.
    The purpose of this function is to provide a default simulation for testing and simplifying examples.

    Example
    -------

    >>> import rumdpy as rp
    >>> import os
    >>> os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS'] = '0'   # Removes warnings from low occupacy (optional)
    >>> import rumdpy as rp
    >>> sim = rp.get_default_sim()

    """
    import rumdpy as rp

    # Setup configuration: FCC Lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    configuration['m'] = 1.0
    temperature = 0.7
    configuration.randomize_velocities(T=temperature)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator: NVT
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup Simulation
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=8,
                        steps_per_timeblock=1024,
                        storage='memory')
    return sim
