import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

include_lammps = True

# List of stored benchmarks to compare with
benchmarks = ['RTX_4090', 'RTX_3070_Laptop', 'RTX_3070_Laptop_LinkedLists']
    
plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for benchmark in benchmarks:
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS']*bdf['N']/1e6, '.-', label=benchmark)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1], '+-', label='Lammps')
plt.legend()
plt.xlim((400, 4e6))
plt.xlabel('N')
plt.ylabel('MATS')
plt.savefig('Data/benhcmarks_mats.pdf')
plt.show()
 