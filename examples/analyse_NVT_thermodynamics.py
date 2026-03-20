""" Compute thermodynamic response functions from an NVT simulation.

Usage:

    python analyse_NVT_thermodynamics.py <filename>

"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import gamdpy as gp

max_plot_points = 100_000

argv = sys.argv.copy()
argv.pop(0)  # remove script name
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename (.h5 added by script)
    else:
        filename = 'Data/LJ_r0.973_T0.70_toread' # Used in testing
else:
    filename = 'Data/LJ_r0.973_T0.70_toread'

output = gp.tools.TrajectoryIO(filename+'.h5').get_h5()
nblocks, nconfs, N, D = output['trajectory/positions'].shape
simbox = output['initial_configuration'].attrs['simbox_data']
V = np.prod(simbox)  # Box volume
first_block = 1
U, W, K = gp.ScalarSaver.extract(output, columns=['U', 'W', 'K'], per_particle=False, first_block=first_block)
dof = D * N - D
data = gp.tools.calculate_response_functions_NVT(N, dof, V, U, W, K)

# Print and write data
to_toml_file = ""
for key in data:
    print(f'{key:>32} = {data[key]:10.5f}')
    to_toml_file += f'{key} = {data[key]}' + '\n'
print(to_toml_file, file=open(filename + '_NVT_thermodynamics.toml', 'w'))
print('Wrote:', filename+'_NVT_thermodynamics.toml')

# Plot fluctuations
times = gp.ScalarSaver.get_times(output, first_block=first_block)

plotindex = range(len(U))
if len(U)>max_plot_points:
    step = int(len(U)/max_plot_points+1)
    plotindex = plotindex[::step]

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
R = np.corrcoef(W, U)[0, 1]
gamma = np.cov(W,U)[0,1]/np.var(U)
axs[0].set_title(f'N={N},  rho={data['density']:.3f},  T_kin={data['kinetic_temperature']:.3f},  P={data['pressure']:.3f},  R={R:.3f},  gamma={gamma:.3f}')
axs[0].set_ylabel('U/N')
axs[1].set_ylabel('W/N')
axs[2].set_ylabel('K/N')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

label  = f'mean: {np.mean(U)/N:.3f}   std: {np.std(U/N):.3f}'
axs[0].plot(times[plotindex], U[plotindex] / N, label=label)
axs[0].axhline(np.mean(U) / N, color='k', linestyle='--')
axs[0].legend(loc=     'upper right')

label  = f'mean: {np.mean(W)/N:.3f}   std: {np.std(W/N):.3f}'
axs[1].plot(times[plotindex], W[plotindex] / N, label=label)
axs[1].axhline(np.mean(W) / N, color='k', linestyle='--')
axs[1].legend(loc=     'upper right')

label  = f'mean: {np.mean(K)/N:.3f}   std: {np.std(K/N):.3f}'
axs[2].plot(times[plotindex], K[plotindex] / N, label=label)
axs[2].axhline(np.mean(K) / N, color='k', linestyle='--')
axs[2].legend(loc=     'upper right')

fig.savefig(filename+'_NVT_thermodynamics.pdf')
print('Wrote:', filename+'_NVT_thermodynamics.pdf')
if __name__ == "__main__":
    plt.show(block=True)
