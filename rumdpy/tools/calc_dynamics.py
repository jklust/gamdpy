import math
import numpy as np
import matplotlib.pyplot as plt

def calc_dynamics_(trajectory, block0, conf_index0, block1, conf_index1, time_index, msd, m4d):
    dR =   trajectory['block'][block1,conf_index1,0,:,:] - trajectory['block'][block0,conf_index0,0,:,:]
    dR += (trajectory['block'][block1,conf_index1,1,:,:] - trajectory['block'][block0,conf_index0,1,:,:])*trajectory.attrs['simbox_initial']
    for i in range(np.max(trajectory['ptype'][:]) + 1):
        msd[time_index,i] += np.mean(np.sum(dR[trajectory['ptype'][:]==i,:]**2, axis=1))
        m4d[time_index,i] += np.mean(np.sum(dR[trajectory['ptype'][:]==i,:]**2, axis=1)**2)

    return msd, m4d

def calc_dynamics(trajectory, first_block):
    
    ptype = trajectory['ptype'][:]
    num_types = np.max(ptype) + 1
    num_blocks, conf_per_block, _, N, D = trajectory['block'].shape

    print(num_types, first_block, num_blocks, conf_per_block, _, N, D)
    
    extra_times = int(math.log2(num_blocks-first_block))-1
    total_times = conf_per_block-1 + extra_times
    count = np.zeros((total_times,1), dtype=np.int32)
    msd = np.zeros((total_times, num_types))
    m4d = np.zeros((total_times, num_types))
    
    times = trajectory.attrs['dt']*2**np.arange(total_times)
    
    for block in range(first_block, num_blocks):
        for i in range(conf_per_block-1):
            count[i] += 1
            calc_dynamics_(trajectory, block, i+1, block, 0, i, msd, m4d)
            
    # Compute times longer than blocks
    for block in range(first_block, num_blocks):
        for i in range(extra_times):
            index = conf_per_block-1 + i 
            other_block = block + 2**(i+1)            
            if other_block < num_blocks:
                count[index] += 1
                calc_dynamics_(trajectory, other_block, 0, block, 0, index, msd, m4d)
 
    msd /= count
    m4d /= count
    alpha2 = 3*m4d/(5*msd**2) - 1 
    return {'times':times, 'msd':msd, 'alpha2':alpha2, 'count':count}

def create_msd_plot(dynamics, figsize=(8,6)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    for dyn in dynamics:
        axs.loglog(dyn['times'], dyn['msd'], '.-', label=dyn['name'])
    axs.set_xlabel('Time')
    axs.set_ylabel('MSD')
    axs.legend()
    return fig, axs

if __name__ == '__main__':
    import sys
    import h5py
    
    argv = sys.argv
    argv.pop(0) # remove name 
    
    first_block = 0
    output_filename = ''
    while argv[0][0]=='-':
        if argv[0] == '-f':
            argv.pop(0)                     # remove '-f'
            first_block = int(argv.pop(0))  # read and remove parameter
        if argv[0] == '-o':
            argv.pop(0)                     # remove '-o'
            output_filename = argv.pop(0)   # read and remove parameter
    
    # The rest should be filenames...
    dynamics = []
    for filename in sys.argv:
        print(filename, ':', end=' ')
        with h5py.File(filename, "r") as f:
            dynamics.append(calc_dynamics(f, first_block))
            dynamics[-1]['name'] = filename[:-3]
    
    fig, axs = create_msd_plot(dynamics)
    if not output_filename=='':
        plt.savefig(output_filename)
    plt.show()
