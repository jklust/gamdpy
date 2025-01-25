import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# List of stored benchmarks to compare with
#benchmarks = ['h100', 'RTX_4090', 'RTX_4070_Laptop', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
benchmarks = ['RTX_2060_Super', 
              'RTX_3070_Laptop',
              #'RTX_4070_Laptop', 
              'RTX_4090_NSq',]
style = ['ro-', 'ro-','bo-', 'bo-', 'r+-','go-', 'go-', 'ko-', 'ko-']

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')


for index, benchmark in enumerate(benchmarks):
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS'], 'o-', label=benchmark)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1]/lammps[:,0]*1e6, '+--', label='Lammps V100')
N = np.array((512, 2e6))
plt.loglog(N, 500 * 1e6 / N, '--', label='MATS=500')
plt.legend()
plt.xlim((400, 1.5e6))
#plt.ylim((1, 1.e6))
plt.xlabel('N')
plt.ylabel('TPS')
plt.savefig('Data/benhcmarks_tps.pdf')
plt.show()
 
