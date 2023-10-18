import numpy as np
import numba
from numba import cuda
import math
import matplotlib.pyplot as plt

##############################################################################
### Stuff we haven't decided where to place yet
#############################################################################

# Define pair-potential. Option to let sympy do it should be ported from rumd4cpu
def LJ_12_6(dist, params):  # LJ: U(r)  =        A12*r**-12 +     A6*r**-6
    A12 = params[0]         #     Um(r) =    -12*A12*r**-13 -   6*A6*r**-6
    A6 = params[1]          #     Umm(r) = 13*12*A12*r**-14 + 7*6*A6*r**-8
    dist_sq = dist*dist     # s = -Um/r =     12*A12*r**-14 +   6*A6*r**-8, Fx = s*dx
    invDist2 = numba.float32(1.0)/dist_sq
    invDist6 = invDist2 * invDist2 * invDist2
    invDist8 = invDist6 * invDist2
    invDist12 = invDist6 * invDist6
    invDist14 = invDist12 * invDist2
    
    u =   numba.float32( 0.5)*(A12*invDist12 +                     A6*invDist6) # Double-counting. Should be elsewhere
    s =   numba.float32( 12.0)*A12*invDist14 + numba.float32( 6.0)*A6*invDist8
    umm = numba.float32(156.0)*A12*invDist14 + numba.float32(42.0)*A6*invDist8
    return u, s, umm # U(r), s == -U'(r)/r, U''(r)

def get_default_compute_plan(configuration):
    """
    Return a default compute_plan (dictionary with a set of parameters specifying how computations are done on the GPU). 
    For now this only depends on the number of particles, and properties of the GPU. but should also depend on cutoff and density
    - 'pb': particle per thread block
    - 'tp': threads per particle
    - 'gridsync': Boolean indicating if syncronization should be done by grid.sync() calls
    - 'skin': used when updating nblist
    - 'UtilizeNIII': Boolean indicating if Newton's third law (NIII) should be utilized (see pairpotential_calculator).
    """
    N = configuration.N
    
    # Get relevant info about the device. At some point we should be able to deal with no device (GPU) available
    device = cuda.get_current_device()
        
    # Apperently we can't ask the device about how many cores it has, neither in total or per SM (Streaming Processor),
    # so we read the latter from a stored dictionary dependent on the compute capability.
    from rumdpy.cc_cores_per_SM_dict import cc_cores_per_SM_dict 
    if device.compute_capability in cc_cores_per_SM_dict:
        cc_cores_per_SM = cc_cores_per_SM_dict[device.compute_capability]
    else:
        print('RUMDPY WARNING: Could not find cc_cores_per_SM for this compute_capability. Guessing: 128')
        cc_cores_per_SM=128
    
    num_SM = device.MULTIPROCESSOR_COUNT
    num_cc_cores = cc_cores_per_SM*num_SM
    warpsize = device.WARP_SIZE

    # pb: particle per (thread) block
    pb = 512
    while N//pb < 2*num_SM: # Performance heuristic 
        pb = pb//2
    if pb<8:
        pb=8
    if pb>256:
        pb=256
   
    # tp: threads per particle
    tp = 1
    while N*tp < 3*num_cc_cores: # Performance heuristic 
        tp += 1
        
    while (pb*tp)%warpsize != 0: # Number of threads per thread-block should be multiplum of warpsize
        tp +=1

    if tp > 16:
        tp = 16
        
    # skin: used when updating nblist
    skin = np.float32(0.5)
    if N > 6*1024:
        skin = np.float32(1.0) # We are (for now) using a N^2 nblist updater, so make the nblist be valid for many steps for large N.

    # UtilizeNIII: Boolean flag indicating if Newton's third law (NIII) should be utilized (see pairpotential_calculator).
    # Utilization of NIII is implemented by using atomic add's to the force array, 
    # so it is inefficient at small system sizes where a lot of conflicts occur.
    UtilizeNIII = True
    if N < 16*1024:
        UtilizeNIII = False

    # gridsync: Bolean flag indicating whether synchronization should be done via grid.sync()
    gridsync = True
    if  N*tp > 4*num_cc_cores: # Heuristic
        gridsync = False

    return {'pb':pb, 'tp':tp, 'skin':skin, 'UtilizeNIII':UtilizeNIII, 'gridsync':gridsync}
        
def plot_scalars(df, N, D, figsize):
    df['e'] = df['u'] + df['k'] # Total energy
    df['Tkin'] =2*df['k']/D/(N-1)
    df['Tconf'] = df['fsq']/df['lap']
    df['du'] = df['u'] - np.mean(df['u'])
    df['de'] = df['e'] - np.mean(df['e'])
    df['dw'] = df['w'] - np.mean(df['w'])
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].plot(df['t'], df['du']/N, '.-', label=f"du/N, var(u)/N={np.var(df['u'])/N:.4}")
    axs[0].plot(df['t'], df['de']/N,  '-', label=f"de/N, var(e)/N={np.var(df['e'])/N:.4}")
    axs[0].set_xlabel('Time')
    axs[0].legend()
    
    axs[1].plot(df['t'], df['Tconf'], '.-', label=f"Tconf, mean={np.mean(df['Tconf']):.3f}")    
    axs[1].plot(df['t'], df['Tkin'], '.-', label=f"Tkin, mean={np.mean(df['Tkin']):.3f}")   
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Temperature')
    axs[1].legend()
 
    R = np.dot(df['dw'], df['du'])/(np.dot(df['dw'], df['dw'])*np.dot(df['du'], df['du']))**0.5
    Gamma = np.dot(df['dw'], df['du'])/(np.dot(df['du'], df['du']))
 
    axs[2].plot(df['u']/N, df['w']/N, '.', label=f"R = {R:.3}")
    axs[2].plot(sorted(df['u']/N), sorted(df['du']/N*Gamma + np.mean(df['w']/N)), 'r--', label=f"Gamma = {Gamma:.3}")
    axs[2].set_xlabel('U/N')
    axs[2].set_ylabel('W/N')
    axs[2].legend()
    plt.show()

    return
