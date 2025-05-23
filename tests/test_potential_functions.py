import numpy as np


def test_potential_functions() -> None:
    import gamdpy as rp
    import numba

    # note: this example assumes these functions were implemented correctly in version bfa77f6e
    assert rp.LJ_12_6(1, [2, 3]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6"
    assert rp.LJ_12_6_sigma_epsilon(1, [2, 3]) == (48384.0, 585216.0, 7635456.0), "Problem with rp.LJ_12_6_sigma_epsilon"
    # rp.LJ_12_6_params_from_sigma_epsilon_cutoff seems not to be used
    #assert rp.LJ_12_6_params_from_sigma_epsilon_cutoff(1, [2, 3, 4]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6_params_from_sigma_epsilon_cutoff"
    # consider moving inner functions out for better testing
    
    # Test make_IPL_n
    for n, r, a in [(12,1,1), (12,2**(1/6),3), (6,2**(1/6),1), (6,2,4), (1,2,3)]:
        ipl_n = rp.make_IPL_n(n)
        expected = (a*r**(-n), n*a*r**(-n-2), a*n*(n+1)*r**(-n-2))
        assert np.all(np.isclose(ipl_n(r, (a,)), expected)), f'Problem with make_IPL_n, {(n,r,a)=}'

    # Test add_potential_functions
    LJ = rp.add_potential_functions(rp.make_IPL_n(12), rp.make_IPL_n(6, first_parameter=1))
    for r, a12, a6 in [(1,1,-1), (2**(1/6),3,-3), (2**(1/6),4,-4), (2,4,4), (2,4,-4)]:
        expected = rp.LJ_12_6(r, (a12, a6))
        assert np.all(np.isclose(LJ(r, (a12,a6)), expected)), f'Problem with  add_potential_functions, {(n,a12,a6)=}'

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

    # Test Hertzian pair potential, u=eps*(1-r/sig)**alpha
    params = 1.0, 2.0, 1.0  # Same as "harmonic repulsion" above
    dist = 0.5
    pot_hertzian = rp.hertzian(dist, params)
    assert np.isclose(pot_hertzian[0],0.25), "Problem with rp.hertzian"
    assert np.isclose(pot_hertzian[1],2.0), "Problem with rp.hertzian"
    assert np.isclose(pot_hertzian[2],2.0), "Problem with rp.hertzian"
    eps, alpha, sig = 1.43, 3.1, 1.24
    r = 0.98
    pot_hertzian_2 = rp.hertzian(r, [eps, alpha, sig])
    assert np.isclose(pot_hertzian_2[0] , eps*(1.0-r/sig)**alpha ), "Problem with rp.hertzian"
    assert np.isclose(pot_hertzian_2[1] , alpha*eps*(1.0-r/sig)**(alpha-1)/sig/r ), "Problem with rp.hertzian"
    assert np.isclose(pot_hertzian_2[2] , eps*alpha*(alpha-1)*(1.0-r/sig)**(alpha-2)/sig/sig ), "Problem with rp.hertzian"




    # needs to add test for apply_shifted_force_cutoff, apply_shifted_potential_cutoff

if __name__ == '__main__':  # pragma: no cover
    test_potential_functions()
