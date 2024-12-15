def test_planar_interactions() -> None:
    import math
    import numpy as np
    import rumdpy as rp

    # Create a configuration object
    rho, T = 1.5, 1.44
    configuration = rp.Configuration(D=3, compute_flags={'lapU':True})
    configuration.make_positions(N=768, rho=rho)
    configuration['m'] = 1.0  
    configuration.randomize_velocities(temperature=T)

    compute_plan = rp.get_default_compute_plan(configuration)

    # Set up a wall
    wall_dist = 6.31 # Ingebrigtsen & Dyre (2014)
    A = 4.0*math.pi/3*rho
    wall_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(9,3))
    potential_params_list   = [[A/15.0, -A/2.0, 3.0], [A/15.0, -A/2.0, 3.0]]              # Ingebrigtsen & Dyre (2014)
    particles_list          = [np.arange(configuration.N), np.arange(configuration.N)]    # All particles feel the walls
    wall_point_list         = [[0, 0, wall_dist/2.0], [0, 0, -wall_dist/2.0] ]
    normal_vector_list      = [[0,0,1],               [0,0,-1]]                            
    walls = rp.setup_planar_interactions(configuration, wall_potential, potential_params_list,
                                        particles_list, wall_point_list, normal_vector_list, compute_plan, verbose=True)

    assert isinstance(walls, dict), "rp.setup_planar_interactions should return a dictionary but has not"
    try: value = walls['interactions']
    except KeyError: print("rp.setup_planar_interactions should have 'interactions' key but it hasn't")
    try: value = walls['interaction_params']
    except KeyError: print("rp.setup_planar_interactions should have 'interaction_params' key but it hasn't")

if __name__ == '__main__':
    test_planar_interactions()
