import numpy as np


def test_potential_functions() -> None:
    import rumdpy as rp
    import numba

    # note: this example assumes these functions were implemented correctly in version bfa77f6e
    assert rp.LJ_12_6(1, [2, 3]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6"
    assert rp.LJ_12_6_sigma_epsilon(1, [2, 3]) == (48384.0, 585216.0, 7635456.0), "Problem with rp.LJ_12_6_sigma_epsilon"
    # rp.LJ_12_6_params_from_sigma_epsilon_cutoff seems not to be used
    #assert rp.LJ_12_6_params_from_sigma_epsilon_cutoff(1, [2, 3, 4]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6_params_from_sigma_epsilon_cutoff"
    # consider moving inner functions out for better testing
    assert rp.harmonic_bond_function(2.5, [2, 100]) == (12.5, -20.0, 100.0), "Problem with rp.harmonic_bond_function"
    # seems correct way: https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-variable-is-a-function
    assert callable(rp.make_IPL_n(12)), "Problem with rp.make_IPL_n"
    from sympy.abc import r,s,e
    potLJ = 4*e*((s/r)**(12)-(s/r)**6)
    potLJ_rp = rp.make_potential_function_from_sympy(potLJ, [s, e])
    assert potLJ_rp(1, [2,3]) == rp.LJ_12_6_sigma_epsilon(1, [2, 3]), "Problem with rp.make_potential_function_from_sympy"

    # Test SAAP potential
    number_of_params = 8
    params = [1.0]*number_of_params
    dist = 1.0
    pot_SAAP = rp.SAAP(dist, params)
    assert len(pot_SAAP) == 3, "Problem with rp.SAAP"

    # Test harmonic repulsion, here u=(1-r)²
    pair_pot = rp.PairPotential(rp.harmonic_repulsion, params=params, max_num_nbs=128)
    params = 2.0, 1.0
    dist = 0.5
    pot_harm_rep = rp.harmonic_repulsion(dist, params)
    assert np.isclose(pot_harm_rep[0],0.25), "Problem with rp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep[1],2.0), "Problem with rp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep[2],2.0), "Problem with rp.harmonic_repulsion"
    eps, sig = 1.43, 1.37
    r = 0.98
    pot_harm_rep_2 = rp.harmonic_repulsion(r, [eps, sig])
    assert np.isclose(pot_harm_rep_2[0], np.float32(0.5*eps*(1.0-r/sig)**2)), f"Problem with rp.harmonic_repulsion"
    du_dr = -eps*(1.0-r/sig)/sig
    assert np.isclose(pot_harm_rep_2[1], -du_dr/r), "Problem with rp.harmonic_repulsion"
    assert np.isclose(pot_harm_rep_2[2], eps/sig**2), "Problem with rp.harmonic_repulsion"

    # needs to add test for apply_shifted_force_cutoff, apply_shifted_potential_cutoff

if __name__ == '__main__':  # pragma: no cover
    test_potential_functions()
