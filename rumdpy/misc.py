import numpy as np
import numba
from numba import cuda
import math
import matplotlib.pyplot as plt
import os

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
    if os.getenv("NUMBA_ENABLE_CUDASIM")!="1": 
        # Trying to handle no device (GPU) case
        # NUMBA_ENABLE_CUDASIM environment variable is set to "1" if the cuda simulator is used.
        # See: https://numba.pydata.org/numba-doc/dev/cuda/simulator.html
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
    else: # Sets up the behaviour in case the GPU simulator is active and set num_cc_cores = number of threads
        num_SM = 1
        num_cc_cores = numba.get_num_threads()
        warpsize = 1

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
    while N*tp < 2*num_cc_cores: # Performance heuristic (conservative) 
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
        
def plot_scalars_old(df, N, D, figsize, block=True):
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
   
    ramp = False
    if 'Ttarget' in df.columns: # Is it a ramp?
        if np.std(df['Ttarget'])>0.01*np.mean(df['Ttarget']):
            ramp = True
            axs[1, 1].plot(df['Ttarget'], df['u']/N, '.-')
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Potenital energy per particle')

    if ramp==False:
        R = np.dot(df['dw'], df['du'])/(np.dot(df['dw'], df['dw'])*np.dot(df['du'], df['du']))**0.5
        Gamma = np.dot(df['dw'], df['du'])/(np.dot(df['du'], df['du']))
 
        axs[1, 1].plot(df['u']/N, df['w']/N, '.', label=f"R = {R:.3}")
        axs[1, 1].plot(sorted(df['u']/N), sorted(df['du']/N*Gamma + np.mean(df['w']/N)), 'r--', label=f"Gamma = {Gamma:.3}")
        axs[1, 1].set_xlabel('U/N')
        axs[1, 1].set_ylabel('W/N')
        axs[1, 1].legend()
        
    plt.show(block=block)

    return

def plot_scalars(df, N, D, figsize, block=True):
    df['E'] = df['U'] + df['K'] # Total energy
    df['Tkin'] =2*df['K']/D/(N-1)
    df['Tconf'] = df['Fsq']/df['lapU']
    df['press'] =  2*df['K']/D/(N-1) * N / df['Vol'] + df['W'] / df['Vol']
    df['dU'] = df['U'] - np.mean(df['U'])
    df['dE'] = df['E'] - np.mean(df['E'])
    df['dW'] = df['W'] - np.mean(df['W'])

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs[0, 0].plot(df['t'], df['dU']/N, '.-', label=f"dU/N, var(U)/N={np.var(df['U'])/N:.4}")
    axs[0, 0].plot(df['t'], df['dE']/N,  '-', label=f"dE/N, var(E)/N={np.var(df['E'])/N:.4}")
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
   
    ramp = False
    if 'Ttarget' in df.columns: # 
        if np.std(df['Ttarget'])>0.01*np.mean(df['Ttarget']):
            ramp = True
            axs[1, 1].plot(df['Ttarget'], df['U']/N, '.-')
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Potenital energy per particle')

    if ramp==False:
        R = np.dot(df['dW'], df['dU'])/(np.dot(df['dW'], df['dW'])*np.dot(df['dU'], df['dU']))**0.5
        Gamma = np.dot(df['dW'], df['dU'])/(np.dot(df['dU'], df['dU']))
 
        axs[1, 1].plot(df['U']/N, df['W']/N, '.', label=f"R = {R:.3}")
        axs[1, 1].plot(sorted(df['U']/N), sorted(df['dU']/N*Gamma + np.mean(df['W']/N)), 'r--', label=f"Gamma = {Gamma:.3}")
        axs[1, 1].set_xlabel('U/N')
        axs[1, 1].set_ylabel('W/N')
        axs[1, 1].legend()
        
    plt.show(block=block)

    return


def get_default_sim():
    """ Return a sim object of the single component LJ crystal in the NVT ensemble.
    The purpose of this function is to provide a default simulation for testing and simplifying examples.
    """
    import rumdpy as rp

    # Setup configuration: FCC Lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    configuration['m'] = 1.0
    temperature = 0.7
    configuration.randomize_velocities(T=temperature)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = rp.PairPotential2(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator: NVT
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup Simulation
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=8,
                        steps_per_timeblock=1024,
                        storage='memory')
    return sim
