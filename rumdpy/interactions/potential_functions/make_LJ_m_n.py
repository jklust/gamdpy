import numpy as np
import numba
import math
from numba import cuda

def make_LJ_m_n(m: float, n: float) -> callable:
    """ Mie Potential

    Also known as the generalized Lennard-Jones potential:

    .. math::

            u(r) = A_m r^{-m} - A_n r^{-n}

    Returns
    -------

    potential_function : callable
        A function that calculates the Mie potential,
        u, s, umm = potential_function(dist, params).
        where params = [A_m, A_n]
    """

    def LJ_m_n(dist, params): # pragma: no cover
        #     U(r) =           Am*r**-m     +         An*r**-n
        #     Um(r) =       -m*Am*r**-(m+1) -       n*An*r**-(n+1)
        #     Umm(r) = (m+1)*m*Am*r**-(m+2) + (n+1)*n*An*r**-(n+2)
        Am = params[0]
        An = params[1]
        invDist = numba.float32(1.0) / dist  #  s = -Um/r =       m*Am*r**-(m+2) +       n*An*r**-(n+2), Fx = s*dx

        u = (Am * invDist ** m + An * invDist ** n)
        s = numba.float32(m) * Am * invDist ** (m + 2) + numba.float32(n) * An * invDist ** (n + 2)
        umm = numba.float32(m * (m + 1)) * Am * invDist ** (m + 2) + numba.float32(n * (n + 1)) * An * invDist ** (
                    n + 2)
        return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

    return LJ_m_n

