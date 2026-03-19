""" Compute thermodynamic response functions from an NpT simulation.

Usage:

    python analyse_NpT_thermodynamics.py <filename>

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
        filename = 'Data/LJ_p4.70_T2.0_toread' # Used in testing
else:
    filename = 'Data/LJ_p4.70_T2.0_toread'

# Read thermodynamic data and compute response functions
output = gp.tools.TrajectoryIO(filename+'.h5').get_h5()
*_, N, D = output['trajectory/positions'].shape
dof = D * N - D
fluctuations = gp.ScalarSaver.extract_as_dict(output)  # Read fluctuation data
response_functions = gp.tools.calculate_response_functions_NpT(N, dof, **fluctuations, k_B=1.0)

# Print and write data
to_toml_file = ""
for key in response_functions:
    print(f'{key:>38} = {response_functions[key]:10.5f}')
    to_toml_file += f'{key} = {response_functions[key]}' + '\n'
print(to_toml_file, file=open(filename + '_NpT_thermodynamics.toml', 'w'))
print('Wrote:', filename+'_NpT_thermodynamics.toml')

# Plot fluctuations
U, W, K, Vol = fluctuations['U'], fluctuations['W'], fluctuations['K'], fluctuations['Vol']
times = gp.ScalarSaver.get_times(output)

plotindex = range(len(U))
if len(U)>max_plot_points:
    step = int(len(U)/max_plot_points+1)
    plotindex = plotindex[::step]

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
R = np.corrcoef(W, U)[0, 1]
gamma = np.cov(W,U)[0,1]/np.var(U)
axs[0].set_title(f'N={N},  rho={response_functions['density']:.3f},  T_kin={response_functions['kinetic_temperature']:.3f},  P={response_functions['pressure']:.3f}')
axs[0].set_ylabel('U/N')
axs[1].set_ylabel('V/N')
axs[2].set_ylabel('K/N')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

label  = f'mean: {np.mean(U)/N:.3f}   std: {np.std(U/N):.3f}'
axs[0].plot(times[plotindex], U[plotindex] / N, label=label)
axs[0].axhline(np.mean(U) / N, color='k', linestyle='--')
axs[0].legend(loc=     'upper right')

label  = f'mean: {np.mean(Vol)/N:.3f}   std: {np.std(Vol/N):.3f}'
axs[1].plot(times[plotindex], Vol[plotindex] / N, label=label)
axs[1].axhline(np.mean(Vol) / N, color='k', linestyle='--')
axs[1].legend(loc=     'upper right')

label  = f'mean: {np.mean(K)/N:.3f}   std: {np.std(K/N):.3f}'
axs[2].plot(times[plotindex], K[plotindex] / N, label=label)
axs[2].axhline(np.mean(K) / N, color='k', linestyle='--')
axs[2].legend(loc=     'upper right')

fig.savefig(filename+'_NpT_thermodynamics.pdf')
print('Wrote:', filename+'_NpT_thermodynamics.pdf')
if __name__ == "__main__":
    plt.show(block=True)
