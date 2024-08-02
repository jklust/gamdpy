""" Plot data generated py isomorph.py

"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

# Load data from pickle file
with open('Data/isomorph.pkl', 'rb') as f:
    data = pickle.load(f)

# Setup figure 
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
axs[0].set_ylabel('RDF')
axs[1].set_ylabel('RDF')
axs[0].set_xlabel('Distance')
axs[1].set_xlabel('Reduced distance')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)

# Loop over simulation (i.e. elements in the list of data)
for simulation in data:
    # Unpack data for convenience
    rho = simulation['rho']
    T = simulation['T']
    distances = simulation['rdf']['distances']
    rdf = np.mean(simulation['rdf']['rdf'], axis=0)

    # Do the actual plotting in absolute and then reduced units
    axs[0].plot(distances, rdf,
                '-', label=f'rho={rho:.3f}, T={T:.3f}')
    axs[1].plot(distances * rho ** (1 / 3), rdf,
                '-', label=f'rho={rho:.3f}, T={T:.3f}')

# Final touches and saving    
axs[0].legend(loc='upper right')
axs[0].set_xlim([0.5, 3.5])
axs[1].set_xlim([0.5, 3.5])
fig.tight_layout()
fig.savefig('isomorph_rdf.pdf')
#  plt.show()
