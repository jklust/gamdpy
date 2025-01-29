import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# List of stored benchmarks to compare with
#benchmarks = ['h100', 'RTX_4090', 'RTX_4070_Laptop', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
benchmarks = [#'RTX_2060_Super_AT', 
              #'RTX_2080_Ti', 
              #'RTX_3070_Laptop_AT',
              'RTX_4070_AT', 
              'RTX_4090_AT',]
style = ['ro', 'bo', 'go', 'ko']

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')

for index, benchmark in enumerate(benchmarks):
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS_AT'],  style[index]+'-', label=benchmark)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1]/lammps[:,0]*1e6, '+-', label='Lammps V100')
N = np.array((512, 2e6))
plt.loglog(N, 500 * 1e6 / N, '-.', label='MATS=500')
plt.legend()
plt.xlim((400, 1.5e6))
plt.ylim((100, 1.e6))
plt.xlabel('N')
plt.ylabel('TPS')
plt.savefig('Data/benhcmarks_tps.pdf')
plt.show(block=False)
 
plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for index, benchmark in enumerate(benchmarks):
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS_AT']*bdf['N']/1e6, style[index]+'-', label=benchmark)
    plt.loglog(bdf['N'], bdf['TPS']*bdf['N']/1e6, style[index]+'--', alpha=0.5)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1], '+-', label='Lammps V100')
plt.legend()
plt.xlim((400, 1.5e6))
#plt.ylim((1, 1.e6))
plt.xlabel('N')
plt.ylabel('MATS')
plt.savefig('Data/benhcmarks_mats.pdf')
plt.show()
 
