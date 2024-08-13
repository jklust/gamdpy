""" Minimal example of a constant NpT simulation

Simulation of a Lennard-Jones liquid in the NPT ensemble.
After equilibration, the simulation runs and calculates the mean potential energy, 
pressure, density and isothermal compressibility. The latter is done by calculating

"""

import numpy as np

import rumdpy as rp

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.7543)
configuration['m'] = 1.0
configuration.randomize_velocities(T=2.0)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup NPT integrator
target_temperature = 2.0  # target temperature for barostat
target_pressure = 4.7  # target pressure for barostat
integrator = rp.integrators.NPT_Atomic(temperature=target_temperature, 
                                       tau=0.4,
                                       pressure=target_pressure, 
                                       tau_p=20, 
                                       dt=0.001)

# NPT Simulation 
sim = rp.Simulation(configuration, pair_pot, integrator,
                    num_timeblocks=16,
                    steps_per_timeblock=2048,
                    steps_between_momentum_reset=100,
                    scalar_output=32,
                    storage='memory')

sim.run()  # Equilibration run
sim.run()  # Production run

# Thermodynamic properties
U, W, K, V = rp.extract_scalars(sim.output, ['U', 'W', 'K', 'Vol'], first_block=1)
print(f'Mean U: {np.mean(U)/configuration.N}')
print(f'Kinetic temperature (consistency check): {2*np.mean(K)/3/(configuration.N-1)}')
print(f'Pressure (consistency check): {(2*np.mean(K)/3+np.mean(W))/np.mean(V)}')
rho = configuration.N/np.mean(V)  # Average density
print(f"Density: {rho}")
compressibility = np.var(V)/np.mean(V)/target_temperature
print(f'Isothermal compressibility: {compressibility}')
print(f'Isothermal bulk modulus: {1/compressibility}')
