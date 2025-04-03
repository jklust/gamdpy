import numpy as np 
import numba 
import math 
from numba import cuda

def cos_angle_function(dr_1, dr_2, params):
    
    f_1 = cuda.local.array(3, dtype=numba.float32)
    f_2 = cuda.local.array(3, dtype=numba.float32)

    kspring, angle = params[0], params[1]

    c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
    c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
    c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]

    cCon = math.cos(math.pi - angle);
    cD = math.sqrt(c11*c22)
    cc = c12/cD 

    f = -kspring*(cc - cCon)
    for k in range(3):
        f_1[k] = f*( (c12/c11)*dr_1[k] - dr_2[k] )/cD
        f_2[k] = f*( dr_1[k] - (c12/c22)*dr_2[k] )/cD

    u = numba.float32(0.5)*kspring*(cc-cCon)*(cc-cCon)

    return f_1, f_2, u 





