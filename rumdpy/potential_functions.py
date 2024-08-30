import numpy as np
import numba
import math
from numba import cuda


# Define pair-potentials.

#  Problem with potential. Maybe becaus it has no parameters.
#def non_interacting(dist, params):
#    """ A pair potential for non-interacting particles (ideal gas or einstein crystal) """
#    return 0.0, 0.0, 0.0

def LJ_12_6(dist, params):  # LJ: U(r)  =        A12*r**-12 +     A6*r**-6
    """ The 12-6 Lennard-Jones potential

    See :func:`rumdpy.apply_shifted_potential_cutoff` for a usage example.

    .. math::

        u(r) = A_{12} r^{-12} + A_6 r^{-6}

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        A₁₂, A₆

    Returns
    -------

    u : float
        Potential energy
    s : float
        Force multiplier, -u'(r)/r
    umm : float
        Second derivative of potential energy

    """
    A12 = params[0]  #     Um(r) =    -12*A12*r**-13 -   6*A6*r**-7
    A6 = params[1]  #     Umm(r) = 13*12*A12*r**-14 + 7*6*A6*r**-8
    invDist = numba.float32(1.0) / dist  # s = -Um/r =     12*A12*r**-14 +   6*A6*r**-8, Fx = s*dx

    u = A12 * invDist ** 12 + A6 * invDist ** 6
    s = numba.float32(12.0) * A12 * invDist ** 14 + numba.float32(6.0) * A6 * invDist ** 8
    umm = numba.float32(156.0) * A12 * invDist ** 14 + numba.float32(42.0) * A6 * invDist ** 8
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)


def LJ_12_6_sigma_epsilon(dist, params):
    """ The 12-6 Lennard-Jones potential
    
    .. math::
    
        u(r) = 4\\epsilon(   (r/\\sigma)^{-12} -   (r/\\sigma)^{-6} )
    
    This is the same as the :func:`rumdpy.LJ_12_6` potential, 
    but with :math:`\\sigma` (sigma) and :math:`\\epsilon` (epsilon) as parameters.
    
    Parameters
    ----------
    
    dist : float
        Distance between particles
        
    params : array-like
        sigma, epsilon

    """  # LJ:  U(r)  =     4*epsilon(   (r/sigma)**-12 +   (r/sigma)**-6 )
    sigma = params[0]  #      Um(r) =   -24*epsilon( 2*(r/sigma)**-13 +   (r/sigma)**-7 )/sigma
    epsilon = params[1]  #      Umm(r) =   24*epsilon(26*(r/sigma)**-14 + 7*(r/sigma)**-8 )/sigma**2
    OneOdist = numba.float32(
        1.0) / dist  # s = -Um/r =     24*epsilon( 2*(r/sigma)**-14 +   (r/sigma)**-8 )/sigma**2,  Fx = s*dx
    sigmaOdist = sigma * OneOdist

    u = numba.float32(4.0) * epsilon * (sigmaOdist ** 12 - sigmaOdist ** 6)
    s = numba.float32(24.0) * epsilon * (numba.float32(2.0) * sigmaOdist ** 12 - sigmaOdist ** 6) * OneOdist ** 2
    umm = numba.float32(24.0) * epsilon * (
                numba.float32(26.0) * sigmaOdist ** 12 - numba.float32(7.0) * sigmaOdist ** 6) * OneOdist ** 2
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)


def LJ_12_6_params_from_sigma_epsilon_cutoff(sigma: float, epsilon: float, cutoff: float) -> np.ndarray:
    """ Convert LJ_12_6_sigma_epsilon (sigma, epsilon, and cutoff) to LJ_12_6 parameters (A12, A6, cutoff).

    Get 'params' array for LJ_12_6 from sigma, epsilon, and cutoff arrays (num_types, num_types)

    .. math::

        4\\epsilon( (\\sigma/r)^{12} - (\\sigma/r)^6) = 4\\epsilon\\sigma^{12}r^{-12} - 4\\epsilon\\sigma^6r^{-6}

    """
    sigma = np.array(sigma, dtype=np.float32)
    epsilon = np.array(epsilon, dtype=np.float32)
    cutoff = np.array(cutoff, dtype=np.float32)

    A12 = 4 * epsilon * sigma ** 12
    A6 = -4 * epsilon * sigma ** 6

    params = np.array([A12, A6, cutoff])
    params = np.moveaxis(params, source=0, destination=2)

    return params


def harmonic_bond_function(dist: float, params: np.ndarray) -> tuple:
    """ Harmonic bond potential

    .. math::

        u(r) = \\frac{1}{2} k (r - r_0)^2

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        r₀, k

    Returns
    -------

    u : float
        Potential energy
    s : float
        Force multiplier, -u'(r)/r
    umm : float
        Second derivative of potential energy

    See Also
    --------

    rumdpy.Bonds

    """
    length = params[0]
    strength = params[1]

    u = numba.float32(0.5) * strength * (dist - length) ** 2
    s = -strength * (dist - length) / dist
    umm = strength
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)


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

    def LJ_m_n(dist, params):  #     U(r) =           Am*r**-m     +         An*r**-n
        Am = params[0]  #     Um(r) =       -m*Am*r**-(m+1) -       n*An*r**-(n+1)
        An = params[1]  #     Umm(r) = (m+1)*m*Am*r**-(m+2) + (n+1)*n*An*r**-(n+2)
        invDist = numba.float32(1.0) / dist  #  s = -Um/r =       m*Am*r**-(m+2) +       n*An*r**-(n+2), Fx = s*dx

        u = (Am * invDist ** m + An * invDist ** n)
        s = numba.float32(m) * Am * invDist ** (m + 2) + numba.float32(n) * An * invDist ** (n + 2)
        umm = numba.float32(m * (m + 1)) * Am * invDist ** (m + 2) + numba.float32(n * (n + 1)) * An * invDist ** (
                    n + 2)
        return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

    return LJ_m_n


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

    def IPL_n(dist, params):  #     U(r) =           An*r**-n
        An = params[0]  #     Um(r) =        n*An*r**-(n+1)
        invDist = numba.float32(1.0) / dist  # s = -Um/r =        n*An*r**-(n+2), Fx = s*dx

        u = An * invDist ** n
        s = numba.float32(n) * An * invDist ** (n + 2)
        umm = numba.float32(n * (n + 1)) * An * invDist ** (n + 2)
        return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

    return IPL_n



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

# Helper functions

def make_potential_function_from_sympy(ufunc, param_names) -> callable:
    """ Make a potential function from a sympy expression

    Known to result in slow code. Use for testing and prototyping.

    Parameters
    ----------

    ufunc : sympy expression
        The potential energy expression in Sympy's symbolic form

    param_names : list
        List of parameters

    Returns
    -------

    potential_function : callable
        A function that calculates the potential energy, force multiplier, and second derivative of the potential energy
        u, s, umm = potential_function(dist, params)

    """
    # import sympy as sp
    # from sympy.utilities.lambdify import lambdify
    # r = sp.symbols('r')

    dufunc = sp.simplify(sp.diff(ufunc, r))  # Sympy functions
    sfunc = sp.simplify(-sp.diff(ufunc, r) / r)
    ummfunc = sp.simplify(sp.diff(dufunc, r))
    u_lam = numba.njit(lambdify([r, param_names], ufunc, 'numpy'))  # Jitted python functions
    s_lam = numba.njit(lambdify([r, param_names], sfunc, 'numpy'))
    umm_lam = numba.njit(lambdify([r, param_names], ummfunc, 'numpy'))

    #@numba.njit
    def potential_function(r, params):
        u = np.float32(u_lam(r, params))
        s = np.float32(s_lam(r, params))
        umm = np.float32(umm_lam(r, params))
        return u, s, umm

    return potential_function


def apply_shifted_potential_cutoff(pair_potential: callable) -> callable:
    """ Apply shifted potential cutoff to a pair-potential function

        If the input pair potential is :math:`u(r)`,
        then the shifted potential is :math:`u(r) - u(r_{c})`, where :math:`r_c` is the cutoff distance.

        Note: calls original potential function twice, avoiding changes to params.

        Parameters
        ----------

        pair_potential : callable
            A function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

        Returns
        -------

        pair_potential : callable
            A function where shifted_potential_cutoff is applied to original function

        Example
        -------

        The following example demonstrates how to use this function to set up the Lenard-Jones 12-6 potential
        truncated and shifted to zero at the cutoff distance of 2.5:

        >>> import rumdpy as rp
        >>> pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6)
        >>> A12, A6, cut = 1.0, 1.0, 2.5
        >>> pair_pot = rp.PairPotential(pair_func, params=[A12, A6, cut], max_num_nbs=1000)

    """
    pair_pot = numba.njit(pair_potential)

    @numba.njit
    def potential(dist, params):
        cut = params[-1]
        u, s, umm = pair_pot(dist, params)
        u_cut, s_cut, umm_cut = pair_pot(cut, params)
        u -= u_cut
        return u, s, umm

    return potential


def apply_shifted_force_cutoff(pair_potential):  # Cut-off by computing potential twice, avoiding changes to params
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
    def potential(dist, params):
        cut = params[-1]
        u, s, umm = pair_pot(dist, params)
        u_cut, s_cut, umm_cut = pair_pot(cut, params)
        u -= u_cut - s_cut * cut * (dist - cut)
        #u -= u_cut - s_cut*dist*(dist-cut)
        s -= s_cut
        return u, s, umm

    return potential
