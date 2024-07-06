""" Test the get_default_sim function. """

def test_get_default_sim():
    import rumdpy as rp
    sim = rp.get_default_sim()
    assert isinstance(sim, rp.Simulation)
    assert isinstance(sim.configuration, rp.Configuration)
    assert sim.configuration['r'] is not None
    assert sim.configuration['m'] is not None
    assert sim.configuration['v'] is not None


if __name__ == '__main__':
    test_get_default_sim()
