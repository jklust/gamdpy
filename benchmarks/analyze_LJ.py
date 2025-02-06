import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# List of stored benchmarks to compare with
#benchmarks = ['h100', 'RTX_4090', 'RTX_4070_Laptop', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
benchmarks = ['RTX_2060_Super_AT', 
              'RTX_2080_Ti_AT', 
              #'RTX_3070_Laptop_AT',
              'RTX_4070_AT', 
              'RTX_4090_AT',]
style = ['ro', 'bo', 'go', 'ko']

# Print benchmarks in markdown
print('## Preliminary benchmarks.')
print()
print('![Fig](./Data/benchmark_LJ_tps.png)')
for index, benchmark in enumerate(benchmarks):
    print('\n' + benchmark+':')
    with open('Data/benchmark_LJ_' + benchmark + '.pkl', 'rb') as file:
        data = pickle.load(file)
    print('|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |')
    print('| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |')
    for n, tps, cp in zip(data['N'], data['TPS_AT'], data['compute_plans_at']):
        print(f"|{n:10} |{tps:8.1f} |{tps*n/1e6:6.1f} |{cp['pb']:4} |{cp['tp']:3} |{cp['skin']:5.2f} |   {cp['gridsync']!s:6} | {cp['nblist']:12} | {cp['UtilizeNIII']!s:5}  |")

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for index, benchmark in enumerate(benchmarks):
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
plt.savefig('Data/benchmark_LJ_tps.pdf')
plt.savefig('Data/benchmark_LJ_tps.png')
plt.show(block=False)
 
plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for index, benchmark in enumerate(benchmarks):
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS_AT']*bdf['N']/1e6, style[index]+'-', label=benchmark)
    plt.loglog(bdf['N'], bdf['TPS']*bdf['N']/1e6, style[index]+'--', alpha=0.5)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1], '+-', label='Lammps V100')
r3 = np.loadtxt('Data/Rumd36_LJ_RTX_2080_Ti.dat')
plt.loglog(r3[:,0],r3[:, 2], '+-', label='Rumd36 RTX_2080_Ti')
plt.legend()
plt.xlim((400, 1.5e6))
#plt.ylim((1, 1.e6))
plt.xlabel('N')
plt.ylabel('MATS')
plt.savefig('Data/benchmark_LJ_mats.pdf')
plt.show(block=False)
 

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for index, benchmark in enumerate(benchmarks):
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS_AT']*bdf['N']/1e6, style[index]+'-', label=benchmark)
    #plt.loglog(bdf['N'], bdf['TPS']*bdf['N']/1e6, style[index]+'--', alpha=0.5)
lammps = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
plt.loglog(lammps[:,0],lammps[:, 1], '+-', label='Lammps V100')
r3 = np.loadtxt('Data/Rumd36_LJ_RTX_2080_Ti.dat')
plt.loglog(r3[:,0],r3[:, 2], style[0]+'--', alpha=0.5, label='Rumd36 RTX_2080_Ti')
r3 = np.loadtxt('Data/Rumd36_LJ_RTX_4090.dat')
plt.loglog(r3[:,0],r3[:, 2], style[2]+'--', alpha=0.5, label='Rumd36 RTX_4090')
plt.legend()
plt.xlim((400, 1.5e6))
#plt.ylim((1, 1.e6))
plt.xlabel('N')
plt.ylabel('MATS')
plt.savefig('Data/benchmarks_mats_rumd3.pdf')
plt.show()
 

