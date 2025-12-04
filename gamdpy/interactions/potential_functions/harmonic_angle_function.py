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

def harmonic_angle_function(theta: float, params: np.ndarray) -> tuple:
    r""" Harmonic angle potential,

    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    Parameters
    ----------

    theta : float
        Angle (radians) defined by three neighboring particles in a molecule.
        Angle zero corresponds to atoms 0, 1, 2 lying consecutively along a straight line, i.e. :math:`\pi` minus the angle subtended by atoms 0 and 2 at 1

    params : array-like
         :math:`\theta_0`, the angle of minimum energy, :math:`k_{spring}`, the spring constant.
        :math:`\theta_0`, is defined differently to angle :math:`\theta`, ie with zero corresponding to zero angle
        subtended by atoms 0 and 2 at 1


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
    d_u_d_cos_theta_neg = kspring * (theta - theta_0) /  (s+numba.float32(0.000001))

    return u, d_u_d_cos_theta_neg

