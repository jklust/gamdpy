import numpy as np
import numba
from numba import cuda
import math
import matplotlib.pyplot as plt

##############################################################################
### Stuff we haven't decided where to place yet
#############################################################################

    
def make_function_constant(value):
    value = np.float32(value)
    def function(x):
        return value
    return function

def make_function_ramp(value0, x0, value1, x1):
    value0, x0, value1, x1 = np.float32(value0), np.float32(x0), np.float32(value1), np.float32(x1)
    alpha = (value1 - value0)/(x1 - x0)
    def function(x):
        if x<x0:
            return value0
        if x<x1:
            return value0 + (x-x0)*alpha
        return value1
    return function
        
def make_function_sin(period, amplitude, offset):
    from math import sin, pi
    period, amplitude, offset = np.float32(period), np.float32(amplitude), np.float32(offset)
    def function(x):
        return offset + amplitude*sin(2*pi*x/period)
    return function
        
    
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
        
def plot_scalars(df, N, D, figsize, block=True):
    df['e'] = df['u'] + df['k'] # Total energy
    df['Tkin'] =2*df['k']/D/(N-1)
    df['Tconf'] = df['fsq']/df['lap']
    df['press'] =  2*df['k']/D/(N-1) * N / df['vol'] + df['w'] / df['vol']
    df['du'] = df['u'] - np.mean(df['u'])
    df['de'] = df['e'] - np.mean(df['e'])
    df['dw'] = df['w'] - np.mean(df['w'])
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].plot(df['t'], df['du']/N, '.-', label=f"du/N, var(u)/N={np.var(df['u'])/N:.4}")
    axs[0, 0].plot(df['t'], df['de']/N,  '-', label=f"de/N, var(e)/N={np.var(df['e'])/N:.4}")
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].legend()
    
    axs[0, 1].plot(df['t'], df['Tconf'], '.-', label=f"Tconf, mean={np.mean(df['Tconf']):.3f}")    
    axs[0, 1].plot(df['t'], df['Tkin'], '.-', label=f"Tkin, mean={np.mean(df['Tkin']):.3f}")   
    if 'Ttarget' in df.columns:
        axs[0, 1].plot(df['t'], df['Ttarget'], 'k--', linewidth=3, label=f"Ttarget,  mean={np.mean(df['Ttarget']):.3f}") 
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Temperature')
    axs[0, 1].legend()
 
    axs[1, 0].plot(df['t'], df['press'], '.-', label=f"press, mean={np.mean(df['press']):.3f}")   
    if 'Ptarget' in df.columns:
        axs[1, 0].plot(df['t'], df['Ptarget'], 'k--', linewidth=3, label=f"Ptarget,  mean={np.mean(df['Ptarget']):.3f}") 

    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure')
    axs[1, 0].legend()
   
    R = np.dot(df['dw'], df['du'])/(np.dot(df['dw'], df['dw'])*np.dot(df['du'], df['du']))**0.5
    Gamma = np.dot(df['dw'], df['du'])/(np.dot(df['du'], df['du']))
 
    axs[1, 1].plot(df['u']/N, df['w']/N, '.', label=f"R = {R:.3}")
    axs[1, 1].plot(sorted(df['u']/N), sorted(df['du']/N*Gamma + np.mean(df['w']/N)), 'r--', label=f"Gamma = {Gamma:.3}")
    axs[1, 1].set_xlabel('U/N')
    axs[1, 1].set_ylabel('W/N')
    axs[1, 1].legend()
    plt.show(block=block)

    return



def normalize_and_save_gr(gr_bins, c1, interaction_params, full_range, steps, filename):

    max_cut = interaction_params[1]
    num_bins = gr_bins.shape[0]
    min_box_dim = min(c1.simbox.data[0], c1.simbox.data[1], c1.simbox.data[2])
    num_gr_updates = steps

    if full_range:
        bin_width = (min_box_dim / 2) / num_bins
    else:
        bin_width = max_cut / num_bins

    # Normalize the g(r) data
    rho = c1.N / (c1.simbox.data[0] * c1.simbox.data[1] * c1.simbox.data[2])
    for i in range(len(gr_bins)):
        r_outer = (i + 1) * bin_width
        r_inner = i * bin_width
        shell_volume = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        expected_num = rho * shell_volume
        gr_bins[i] /= (expected_num * num_gr_updates * c1.N)

  
    # Save data to file
    distances = np.arange(0, len(gr_bins)) * bin_width
    data_to_save = np.column_stack((distances, gr_bins))
    np.savetxt(filename, data_to_save, comments='', fmt='%f')
 
    return data_to_save


