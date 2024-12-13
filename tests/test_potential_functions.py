def test_potential_functions() -> None:
    import rumdpy as rp
    # note: this example assumes these functions were implemented correctly in version bfa77f6e
    assert rp.LJ_12_6(1, [2, 3]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6"
    assert rp.LJ_12_6_sigma_epsilon(1, [2, 3]) == (48384.0, 585216.0, 7635456.0), "Problem with rp.LJ_12_6_sigma_epsilon"
    # rp.LJ_12_6_params_from_sigma_epsilon_cutoff seems not to be used
    #assert rp.LJ_12_6_params_from_sigma_epsilon_cutoff(1, [2, 3, 4]) == (5.0, 42.0, 438.0), "Problem with rp.LJ_12_6_params_from_sigma_epsilon_cutoff"
    # needs to add test for apply_shifted_force_cutoff, apply_shifted_potential_cutoff, make_potential_function_from_sympy
    # consider moving inner functions out for better testing
    assert rp.harmonic_bond_function(2.5, [2, 100]) == (12.5, -20.0, 100.0), "Problem with rp.harmonic_bond_function"
    # seems correct way: https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-variable-is-a-function
    assert callable(rp.make_IPL_n(12)), "Problem with rp.make_IPL_n"
    from sympy.abc import r,s,e
    potLJ = 4*e*((s/r)**(12)-(s/r)**6)
    potLJ_rp = rp.make_potential_function_from_sympy(potLJ, [s, e])
    assert potLJ_rp(1, [2,3]) == rp.LJ_12_6_sigma_epsilon(1, [2, 3]), "Problem with rp.make_potential_function_from_sympy"
    # The function rp.SAAP is not fully implemented: expression exp is not defined, should be numba.exp, numpy.exp or math.exp?
    #assert rp.SAAP(), "Problem with rp.SAAP"
    # make_potential_function_from_sympy needs a test

if __name__ == '__main__':
    test_potential_functions()
