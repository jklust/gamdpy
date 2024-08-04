""" Minimal example of a Simulation using rumdpy.

Simulation of a Lennard-Jones liquid in the NPT ensemble.
It`s possible to switch between Langevin and Atomic NPT integrators.

"""

import numpy as np

import rumdpy as rp

# Here you can decide to use "NPT_Atomic" or "NPT_Langevin"
flag = "Atomic"
my_T, my_rho, my_p = 2.0, 0.754289412611, 4.7     # Pressure should be P=4.7 for T=2.0 at this density

# Choose integrator
#match flag:
#    case "Atomic"  : integrator = rp.integrators.NPT_Atomic  (temperature=my_T, tau=0.4, pressure=my_p, tau_p=20, dt=0.001)
#    case "Langevin": integrator = rp.integrators.NPT_Langevin(temperature=my_T, pressure=my_p, alpha=TODO, alpha_baro=TODO, mass_baro=TODO,
#                                                              volume_velocity=TODO, barostatModeISO=True, boxFlucCoord=TODO, dt=0.001, seed=TODO)

if flag=="Atomic":
    integrator = rp.integrators.NPT_Atomic  (temperature=my_T, tau=0.4, pressure=my_p, tau_p=20, dt=0.001)
elif flag=="Langevin":
    rp.integrators.NPT_Langevin(temperature=my_T, pressure=my_p, alpha=TODO, alpha_baro=TODO, mass_baro=TODO,
                                                             volume_velocity=TODO, barostatModeISO=True, boxFlucCoord=TODO, dt=0.001, seed=TODO)


print(f"\nRunning an NPT simulation using the integrator NPT_{flag} at (P, T) = ({my_p}, {my_T})\n")
# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=my_rho) 
configuration['m'] = 1.0
configuration.randomize_velocities(T=my_T)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# NPT Simulation 
sim = rp.Simulation(configuration, pair_pot, integrator,
                    num_timeblocks=16, steps_per_timeblock=2048,
                    steps_between_momentum_reset=100, scalar_output=32,
                    storage='memory')

# Equilibration
print("Equilibration run")
sim.run()
# Data
print("Data run")
sim.run()

U, W, K, Vol = rp.extract_scalars(sim.output, ['U', 'W', 'K', 'Vol'], first_block=1)
print(f"NPT simulation at (P, T) = ({my_p}, {my_T})")
print(f"Mean values  of U, W and T_kin: {np.mean(U)/configuration.N} {np.mean(W)/configuration.N} {2*np.mean(K)/3/configuration.N}")
print(f"Standard dev of U, W and T_kin: {np.std(U)/configuration.N} {np.std(W)/configuration.N} {2*np.std(K)/3/configuration.N}")
print(f"Pressure (mean): {(2*np.mean(K)/3+np.mean(W))/np.mean(Vol)}") 
print(f"Mean and std of Volume: {np.mean(Vol)} {np.std(Vol)}")
print(f"Density should be {my_rho} and is {configuration.N/np.mean(Vol)}")

