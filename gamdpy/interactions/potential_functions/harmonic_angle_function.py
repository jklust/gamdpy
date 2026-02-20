#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 08:55:28 2025

@author: nbailey
"""

import numpy as np
import numba
import math
from numba import cuda




def make_harmonic_angle_function(SMALL=1.e-6):
    """
    Create a version of harmonic_angle_function with a specified value of the regularization
    parameter SMALL. This prevents overflow when dividing by small values of sin(theta).
    """
    def harmonic_angle_function(theta: float, params: np.ndarray) -> tuple:
        theta_0 = params[0]
        kspring = params[1]
        s = math.sin(theta)
        u = numba.float32(0.5) * kspring * (theta - theta_0) ** 2
        d_u_d_cos_theta_neg = kspring * (theta - theta_0) /  (s+SMALL)
        return u, d_u_d_cos_theta_neg
    return harmonic_angle_function


def harmonic_angle_function(theta: float, params: np.ndarray) -> tuple:
    r""" Harmonic angle potential,

    Original version but with a regularization parameter SMALL hard-coded in.
    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    Parameters
    ----------

    theta : float
        Angle (radians) defined by three neighboring particles in a molecule, more precisely: the angle subtended by atoms 0 and 2 at 1

    params : array-like
         :math:`\theta_0`, the angle of minimum energy, :math:`k_{spring}`, the spring constant.

    Returns
    -------

    u : float
        Potential energy
    d_u_cos_theta_neg: float
        Negative derivative of the potential energy with respect to cos(theta)

    See Also
    --------

    gamdpy.Angles

    """

    theta_0 = params[0]
    kspring = params[1]
    s = math.sin(theta)
    u = numba.float32(0.5) * kspring * (theta - theta_0) ** 2
    #d_u_d_cos_theta_neg = kspring * (theta - theta_0) / s
    # To protect against dividing by zero, inspired by LAMMPS:
    SMALL = numba.float32(1.e-6)
    d_u_d_cos_theta_neg = kspring * (theta - theta_0) /  (s+SMALL)

    return u, d_u_d_cos_theta_neg

