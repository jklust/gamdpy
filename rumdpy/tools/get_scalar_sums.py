def get_scalar_sums(self, first_block=0):
    """ Return a dictionary with summed scalar particle data summed for each timeblock.
    The input first_block is the first block to include in the output.

    Returns a dictionary with keys like 't', 'u', 'w', 'lapU', 'fsq', 'k', 'vol' ...
    The first key, 't', is the time for each timeblock.

    Examples:

    >>> import rumdpy as rp
    >>> sim = rp.get_default_sim()
    >>> sim.run(verbose=False)
    >>> data = rp.tools.get_scalar_sums(sim)  # Thermodynamic data in a dictionary
    >>> print(data.keys())
    dict_keys(['t', 'u', 'w', 'lap', 'm', 'k', 'fsq'])
    """
    import numpy as np
    columns = self.configuration.sid.keys()
    data = np.array(self.scalars_list[first_block:])
    time_blocks, _, _ = data.shape
    # Sum over particles
    data = data.sum(axis=1)
    output = dict(zip(columns, data.T))
    # Add time for each block
    output['t'] = np.arange(time_blocks) * self.dt * self.steps_per_block
    # Put t in front of the dictionary
    output = {key: output[key] for key in ['t', *columns]}
    return output
