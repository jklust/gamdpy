import numba
import numpy as np

def add_potential_functions(potential_1, potential_2):  
    """ Add two potential functions into a single potential function
        Note: the two potential functions will have the same cut-off,
              by convention allways stored as last entry in params.
        Note: potential_1 can NOT explicitly depend on cut-off (last entry in params)
        Note: potential_2 should know where to 'look' for its parameters, 
        see  first_parameter in make_IPL_n


    Parameters
    ----------
        potential_1: callable
            a function that calculates a pair-potential:
            u, s, umm =  potential_1(dist, params)
        potential_2: callable
            a function that calculates a pair-potential:
            u, s, umm =  potential_2(dist, params)
    Returns
    -------

        potential: callable
            a function implementing the sum of potential_1 and potential_2

    """
    pair_pot1 = numba.njit(potential_1)
    pair_pot2 = numba.njit(potential_2)

    #@numba.njit
    def potential(dist, params): # pragma: no cover
        u1, s1, umm1 = pair_pot1(dist, params)
        u2, s2, umm2 = pair_pot2(dist, params)
        return u1+u2, s1+s2, umm1+umm2

    return potential

