""" This example demonstrates how thermodynamic properties can exatracted from the simulation object.

This examples first run a (default) simulation and then extracts the thermodynamic data from the simulation object.
This is done using the `get_scalar_sums` method of the simulation object.

The method `get_scalar_sums` returns a dictionary with the sum of particle scalars
such as potential energy, kinetic energy, virial, etc.
The thermodynamic data can be stored in a pandas DataFrame for further analysis.

"""

import rumdpy as rp
import matplotlib.pyplot as plt
import pandas as pd

# Setup and run default simulation (NVT simulation of a Lennard-Jones fcc crystal)
sim = rp.get_default_sim()
sim.run()

# Extract thermodynamic data after each timeblock
thermo_data = rp.tools.get_scalar_sums(sim)  # Thermodynamic data in a dictionary
print(f'{thermo_data.keys()=}')

# Plot the potential energy as a function of time
plt.figure()
plt.plot(thermo_data['t'], thermo_data['u'], 'o')
plt.xlabel('Time')
plt.ylabel('Potential energy')
plt.show()

# Convert data to a pandas DataFrame
df = pd.DataFrame(thermo_data)  # Convert dictionary to a pandas DataFrame
print(f'{df.columns=}')

# Mean potential energy per particle averaged over the simulation
N = sim.configuration.N  # Number of particles
print(f"Number of particles: {N}")
print(f"Mean potential energy per particle: {df['u'].mean()/N}")

# Compute mean pressure (from virial, W)
L = sim.configuration.simbox.lengths  # Box lengths
V = L[0]*L[1]*L[2]  # Volume
T = sim.integrator.temperature  # Temperature
df['p'] = (N*T+df['w'])/V  # p = (NkT + W)/V  where W is the virial
print(f"Mean pressure: {df['p'].mean()}")

# Print summary statistics
print(df.describe())
