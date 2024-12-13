import numpy as np
import numba
import math
from numba import cuda

def apply_shifted_force_cutoff(pair_potential):  
    # Cut-off by computing potential twice, avoiding changes to params
    """ Apply shifted force cutoff to a pair-potential function

    If the input pair potential is :math:`u(r)`, then the shifted force potential is
    :math:`u(r) - u(r_{c}) + s(r_{c})(r - r_{c})`, where :math:`r_c` is the cutoff distance,
    and :math:`s(r) = -u'(r)/r`.


    Note: calls original potential function  twice, avoiding changes to params

    Parameters
    ----------
        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where shifted force cutoff is applied to original function

    """
    pair_pot = numba.njit(pair_potential)

    @numba.njit
    def potential(dist, params): # pragma: no cover
        cut = params[-1]
        u, s, umm = pair_pot(dist, params)
        u_cut, s_cut, umm_cut = pair_pot(cut, params)
        u -= u_cut - s_cut * cut * (dist - cut)
        #u -= u_cut - s_cut*dist*(dist-cut)
        s -= s_cut
        return u, s, umm

    return potential

