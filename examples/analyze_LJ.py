import rumdpy as rp
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt

with open('Data/LJ_pdict.pkl', 'rb') as f:
    pdict = pickle.load(f)
print('pdict: ', pdict)

if pdict['rdf']:
    rdf = np.loadtxt('Data/LJ_rdf.dat')
    plt.figure()
    plt.plot(rdf[:,0], rdf[:,1], '-')
    plt.xlabel('distance')
    plt.ylabel('Radial distribution function')
    plt.show(block=False)

df = pd.read_csv('Data/LJ_scalars.csv')
rp.plot_scalars(df, pdict['N'],  pdict['D'], figsize=(10,8), block=True)

