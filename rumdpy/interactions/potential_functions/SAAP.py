import numpy as np
import numba
import math
from math import exp
from numba import cuda

def SAAP(dist, params):
    '''
    The SAAP potential: u(r) = eps (a0 exp(a1 r)/r + a2 exp(a3 r) + a4) / (1 + a5 r⁶)

    parameters: a0, a1, a2, a3, a4, a5, sigma, eps

    The SAAP potential is a pair potential for noble elements, its parameters
    are derived from fitting on data obtained from ab initio methods (coupled cluster).
    The potential is given by:

        u(r) = eps (a0 exp(a1 r)/r + a2 exp(a3 r) + a4) / (1 + a5 r⁶)

    The s(r) function, used to compute pair forces (𝐅=s·𝐫), is defined as

        s(r) = -u'(r)/r

    Together with the second derivative of u ('d2u_dr2') has been computed
    using the sympy library.

    NB: the six parameters 'ai' are given in units of eps and sigma, e.g. a0 and
    a1 have units of energy*distance and 1/distance respectively (real a0, namely
    A0, is A0 = a0 * sigma * eps).

    '''

    # Extract parameters compatibly with numba, in float32 precision
    a0 = numba.float32(params[0])
    a1 = numba.float32(params[1])
    a2 = numba.float32(params[2])
    a3 = numba.float32(params[3])
    a4 = numba.float32(params[4])
    a5 = numba.float32(params[5])
    sigma = numba.float32(params[6])
    eps = numba.float32(params[7])

    # Definition of a reduced distance to make all consistent with the fact that
    # the various a parameters are given in units of eps and sigma
    r = dist / sigma
    # Compute helper variables
    one = numba.float32(1.0)
    inv_r = one/r

    exp_a1_r = exp(a1 * r)
    a0_exp_r = a0 * exp_a1_r * inv_r  # a0 exp(a1 r)/r
    a2_exp = a2 * exp(a3 * r)
    inverse_one_a5 = one / (one + a5 * r**6)

    # Compute pair potential energy, pair force and pair curvature

    # SAAP computation for r
    u = eps * (a0_exp_r + a2_exp + a4) * inverse_one_a5

    # Compute helper variables for s
    s1 = -6*a5*r**5*(a0*exp(a1*r)/r + a2*exp(a3*r) + a4)/(a5*r**6 + one)**2
    s2 = (a0*a1*exp(a1*r)/r - a0*exp(a1*r)/r**2 + a2*a3*exp(a3*r))/(a5*r**6 + one)
    # First derivative of SAAP divided by r with a minus sign
    s = - eps * (s1 + s2) / (r * sigma**2) # sigma² because of chain rule (CR) and 1/dist = 1/(r sigma)

    # Compute helper variables for u''(r)
    d2u_dr2_1 = numba.float32(72.0)*a5**2*r**10*(a0*exp(a1*r)/r
                                   + a2*exp(a3*r)
                                   + a4)/(a5*r**6 + one)**3
    d2u_dr2_2 = -numba.float32(12.0)*a5*r**5*(a0*a1*exp(a1*r)/r
                                - a0*exp(a1*r)/r**2
                                + a2*a3*exp(a3*r))/(a5*r**6 + one)**2
    d2u_dr2_3 = -numba.float32(30.0)*a5*r**4*(a0*exp(a1*r)/r
                                + a2*exp(a3*r)
                                + a4)/(a5*r**6 + one)**2
    d2u_dr2_4 = (a0*a1**2*exp(a1*r)/r
                 - numba.float32(2.0)*a0*a1*exp(a1*r)/r**2
                 + numba.float32(2.0)*a0*exp(a1*r)/r**3
                 + a2*a3**2*exp(a3*r))/(a5*r**6 + one)
    # Second derivative of SAAP
    d2u_dr2 = eps * (d2u_dr2_1 + d2u_dr2_2 + d2u_dr2_3 + d2u_dr2_4) / sigma**2 # sigma² because of double CR

    return u, s, d2u_dr2  # u(r), -u'(r)/r, u''(r)

