import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# List of stored benchmarks to compare with
#benchmarks = ['h100', 'RTX_4090', 'RTX_4070_Laptop', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
benchmarks = ['RTX_4090', 'RTX_4090_LinkedLists', 'RTX_3070_Laptop', 'RTX_3070_Laptop_LinkedLists']

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for benchmark in benchmarks:
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS'], '.-', label=benchmark)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1]/lammps[:,0]*1e6, '+-', label='Lammps')
plt.loglog(bdf['N'], 400 * 1e6 / bdf['N'], '--', label='Perfect scaling (MATS=400)')
plt.legend()
plt.xlim((400, 4e6))
plt.xlabel('N')
plt.ylabel('TPS')
plt.savefig('Data/benhcmarks_tps.pdf')
plt.show()
 