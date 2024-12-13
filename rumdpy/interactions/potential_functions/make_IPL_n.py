import numpy as np
import numba
import math
from numba import cuda

def make_IPL_n(n: float) -> callable:
    """ Inverse Power Law Potential

    .. math::

        u(r) = A_n r^{-n}

    Parameters
    ----------

    n : float
        Exponent in the potential

    Returns
    -------

    potential_function : callable
        A function that calculates the IPL potential,
        u, s, umm = potential_function(dist, params).
        where params = [A_n]
    """

    def IPL_n(dist, params):  # pragma: no cover
        #     U(r) =           An*r**-n
        #     Um(r) =        n*An*r**-(n+1)
        # s = -Um/r =        n*An*r**-(n+2), Fx = s*dx
        An = params[0]
        invDist = numba.float32(1.0) / dist

        u = An * invDist ** n
        s = numba.float32(n) * An * invDist ** (n + 2)
        umm = numba.float32(n * (n + 1)) * An * invDist ** (n + 2)
        return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

    return IPL_n

