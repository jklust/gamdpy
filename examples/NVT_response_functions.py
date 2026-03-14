""" Compute thermodynamic response functions from an NVT simulation.

Usage:

    python NVT_response_functions.py <filename>

"""

import sys

import numpy as np
import gamdpy as gp

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
U, W, K = gp.ScalarSaver.extract(output, columns=['U', 'W', 'K'], per_particle=False, first_block=1)
dof = D * N - D
data = gp.tools.get_NVT_response_functions(N, dof, V, U, W, K)

#print(data)
for key in data:
    print(f'{key:>32} : {data[key]:10.5f}')
