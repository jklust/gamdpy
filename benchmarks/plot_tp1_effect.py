import matplotlib.pyplot as plt
import pandas as pd

# List of stored benchmarks to compare with
benchmarks = ['RTX_4090', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
    
plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for benchmark in benchmarks:
    print(benchmark)
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    bdf_tp1 = pd.read_csv('Data/benchmark_LJ_' + benchmark + '_tp1.csv')
    plt.semilogx(bdf['N'], bdf['TPS']/bdf_tp1['TPS'], '.-', label=benchmark)
plt.legend()
plt.xlabel('N')
plt.ylabel('TPS / TPS(tp=1)')
plt.savefig('Data/benhcmarks_tp1_effect.pdf')
plt.show()
 