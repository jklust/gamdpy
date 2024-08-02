""" Plot data generated py isomorph.py

"""

import pickle

import matplotlib.pyplot as plt

# Load data from pickle file
with open('Data/isomorph.pkl','rb') as f:
    data = pickle.load(f)

# Setup figure 
fig, axs = plt.subplots(3, 1, figsize=(8,9), sharex=True)
fig.subplots_adjust(hspace=0.00) # Remove vertical space between axes
axs[0].set_ylabel('Reduced MSD')
axs[1].set_ylabel('Non Gaussian parameter')
axs[2].set_ylabel('Intermediate scattering function')
axs[2].set_xlabel('Reduced Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

# Loop over simulation (i.e. elements in the list of data)
for simulation in data:
    
    # Unpack data for convenience
    rho = simulation['rho']
    T = simulation['T']
    dynamics = simulation['dynamics']

    # Do the actual plotting in reduced units
    axs[0].loglog(dynamics['times']*rho**(1/3)*float(T)**0.5, dynamics['msd']*rho**(2/3), 
               'o--', label=f'rho={rho:.3f}, T={T:.3f}')
    axs[1].semilogx(dynamics['times']*rho**(1/3)*float(T)**0.5, dynamics['alpha2'], 
               'o--', label=f'rho={rho:.3f}, T={T:.3f}')
    axs[2].semilogx(dynamics['times']*rho**(1/3)*float(T)**0.5, dynamics['Fs'], 
               'o--', label=f'rho={rho:.3f}, T={T:.3f}, q={dynamics["qvalues"][0]:.3f}')
    
# Final touches and saving    
axs[0].loglog(dynamics['times'][:8], 3*dynamics['times'][:8]**2,
           'k-', label=f'Ballistic prediction', alpha=0.5)
axs[0].legend(loc='upper left')
axs[2].legend()

fig.tight_layout()
#fig.subplots_adjust(hspace=0.00) # Remove vertical space between axes
fig.savefig('isomorph_dynamics.pdf')
#plt.show()
