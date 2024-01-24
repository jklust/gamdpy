import numpy as np
import numba
import math
from numba import cuda

# Define pair-potentials.

def LJ_12_6(dist, params):            # LJ: U(r)  =        A12*r**-12 +     A6*r**-6
    A12 = params[0]                   #     Um(r) =    -12*A12*r**-13 -   6*A6*r**-6
    A6 = params[1]                    #     Umm(r) = 13*12*A12*r**-14 + 7*6*A6*r**-8
    invDist = numba.float32(1.0)/dist # s = -Um/r =     12*A12*r**-14 +   6*A6*r**-8, Fx = s*dx

    u =                        A12*invDist**12 +                     A6*invDist**6 
    s =   numba.float32( 12.0)*A12*invDist**14 + numba.float32( 6.0)*A6*invDist**8
    umm = numba.float32(156.0)*A12*invDist**14 + numba.float32(42.0)*A6*invDist**8
    return u, s, umm # U(r), s == -U'(r)/r, U''(r)


def LJ_12_6_sigma_epsilon(dist, params):  # LJ:  U(r)  =     4*epsilon(   (r/sigma)**-12 +   (r/sigma)**-6 )
    sigma = params[0]                     #      Um(r) =   -24*epsilon( 2*(r/sigma)**-13 +   (r/sigma)**-7 )/sigma
    epsilon = params[1]                   #      Umm(r) =   24*epsilon(26*(r/sigma)**-14 + 7*(r/sigma)**-8 )/sigma**2
    OneOdist = numba.float32(1.0)/dist    # s = -Um/r =     24*epsilon( 2*(r/sigma)**-14 +   (r/sigma)**-8 )/sigma**2,  Fx = s*dx
    sigmaOdist = sigma*OneOdist

    u =   numba.float32( 4.0)*(                     sigmaOdist**12 +                    sigmaOdist**6 )
    s =   numba.float32(24.0)*( numba.float32( 2.0)*sigmaOdist**12 +                    sigmaOdist**6 )*OneOdist**2
    umm = numba.float32(24.0)*( numba.float32(26.0)*sigmaOdist**12 + numba.float32(7.0)*sigmaOdist**6 )*OneOdist**2
    return u, s, umm # U(r), s == -U'(r)/r, U''(r)


def LJ_12_6_params_from_sigma_epsilon_cutoff(sigma, epsilon, cutoff):
    """
    Get 'params' array for LJ_12_6 from sigma, epsilon, and cutoff arrays (num_types, num_types)
    LJ = 4*epsilon*( (sigma/r)**12 - (sigma/r)**6) = 4*epsilon*sigma**12*r**-12 - 4*epsilon*sigma**6*r**-6
    """
    sigma = np.array(sigma, dtype=np.float32)
    epsilon = np.array(epsilon, dtype=np.float32)
    cutoff = np.array(cutoff, dtype=np.float32)
    
    A12 = 4*epsilon*sigma**12
    A6 = -4*epsilon*sigma**6
    
    params = np.array([A12, A6, cutoff])
    params = np.moveaxis(params, source=0, destination=2)
    
    return params

def make_LJ_m_n(m, n):                   
    def LJ_m_n(dist, params):             #     U(r) =           Am*r**-m     +         An*r**-n
        Am = params[0]                    #     Um(r) =       -m*Am*r**-(m+1) -       n*An*r**-(n+1)
        An = params[1]                    #     Umm(r) = (m+1)*m*Am*r**-(m+2) + (n+1)*n*An*r**-(n+2)
        invDist = numba.float32(1.0)/dist #  s = -Um/r =       m*Am*r**-(m+2) +       n*An*r**-(n+2), Fx = s*dx

        u =                           (Am*invDist**m     +                          An*invDist**n) 
        s =   numba.float32( m ) *     Am*invDist**(m+2) + numba.float32( n ) *     An*invDist**(n+2)
        umm = numba.float32( m*(m+1) )*Am*invDist**(m+2) + numba.float32( n*(n+1) )*An*invDist**(n+2)
        return u, s, umm # U(r), s == -U'(r)/r, U''(r)
    return LJ_m_n

def make_IPL_n(n):                   
    def IPL_n(dist, params):              #     U(r) =           An*r**-n
        An = params[0]                    #     Um(r) =        n*An*r**-(n+1)
        invDist = numba.float32(1.0)/dist # s = -Um/r =        n*An*r**-(n+2), Fx = s*dx

        u =                            An*invDist**n 
        s =   numba.float32( n ) *     An*invDist**(n+2)
        umm = numba.float32( n*(n+1) )*An*invDist**(n+2)
        return u, s, umm # U(r), s == -U'(r)/r, U''(r)
    return IPL_n

def harmonic_bond_function(dist, params):
    length = params[0]
    strength = params[1]
    
    u = numba.float32( 0.5)*strength*(dist-length)**2
    s = -strength*(dist-length)/dist
    umm = strength
    return u, s, umm # U(r), s == -U'(r)/r, U''(r)    


# Helper functions

def make_potential_function_from_sympy(ufunc, param_names): 
    dufunc = sp.simplify(sp.diff(ufunc, r))                         # Sympy functions
    sfunc = sp.simplify(-sp.diff(ufunc, r)/r)
    ummfunc = sp.simplify(sp.diff(dufunc, r))
    u_lam = numba.njit(lambdify([r, param_names], ufunc, 'numpy'))  # Jitted python functions
    s_lam = numba.njit(lambdify([r, param_names], sfunc, 'numpy'))
    umm_lam = numba.njit(lambdify([r, param_names], ummfunc, 'numpy'))
   
    #@numba.njit
    def potential_function(r, params):
        u = np.float32(u_lam(r, params))
        s = np.float32(s_lam(r, params))
        umm =  np.float32(umm_lam(r, params))
        return u, s, umm

    return potential_function

def apply_shifted_potential_cutoff(pairpotential):  
    """
        Input:
            pairpotential: a function that calculates a pair-potential
                            u, s,  umm =  pairpotential(dist, params)
        Returns:
            potential: a function where shifted_potential_cutoff is applied to original function
                        (calls original potential function twice, avoiding changes to params)
    """ 
    pairpotential = numba.njit(pairpotential)
    @numba.njit
    def potential(dist, params):
        cut = params[-1]
        u,     s,     umm =     pairpotential(dist, params)
        u_cut, s_cut, umm_cut = pairpotential(cut,  params)
        u -= u_cut
        return u, s, umm
    return potential

def apply_shifted_force_cutoff(pairpotential):  # Cut-off by computing potential twice, avoiding changes to params
    """
        Input:
            pairpotential: a function that calculates a pair-potential
                            u, s,  umm =  pairpotential(dist, params)
        Returns:
            potential: a function where shifted force cutoff is applied to original function
                        (calls original potential function  twice, avoiding changes to params)
    """ 
    pairpotential = numba.njit(pairpotential)
    @numba.njit
    def potential(dist, params):
        cut = params[-1]
        u,     s,     umm =     pairpotential(dist, params)
        u_cut, s_cut, umm_cut = pairpotential(cut,  params)
        u -= u_cut - s_cut*dist*(dist-cut) 
        s -= s_cut
        return u, s, umm
    return potential
