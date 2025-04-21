import numpy as np
import numba
from numba import cuda
import math

def make_function_constant(value):
    """ Return a function that returns a constant value """
    value = np.float32(value)

    def function(x):
        return value

    return function


def make_function_ramp(value0, x0, value1, x1):
    """ Return a function that ramps linearly between two values

    Return a function that returns a constant value for x<x0,
    linearly ramps from value0 to value1 for x0<=x<=x1,
    and returns value1 for x>x1.

    """
    value0, x0, value1, x1 = np.float32(value0), np.float32(x0), np.float32(value1), np.float32(x1)
    alpha = (value1 - value0) / (x1 - x0)

    def function(x):
        if x < x0:
            return value0
        if x < x1:
            return value0 + (x - x0) * alpha
        return value1

    return function


def make_function_sin(period, amplitude, offset):
    """ Return a function that returns a sin function with given period, amplitude and offset """

    from math import sin, pi
    period, amplitude, offset = np.float32(period), np.float32(amplitude), np.float32(offset)

    def function(x):
        return offset + amplitude * sin(2 * pi * x / period)

    return function


