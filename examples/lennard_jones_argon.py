"""  Lennard-Jones conversion from Argon units to SI units

The simulation is done in reduced units where σ=ε=m=1.
This script contains a dictionary that can be used to generate conversion factors,
and show how to convert between reduced and SI units.

"""
from pprint import pprint

import numpy as np
from scipy.constants import Boltzmann, atomic_mass, Avogadro

import gamdpy as gp

# Compute Argon conversion factors (from reduced units to SI units)
print(f'{Boltzmann = }, {atomic_mass = }, {Avogadro = }')
sigma_in_Angstrom = 3.4  # Ångstrom
epsilon_in_K = 120  # K
mass_in_u = 39.948  # Dalton (aka u or amu)
sigma = sigma_in_Angstrom * 1e-10  # m
epsilon = epsilon_in_K*Boltzmann  # J
mass = mass_in_u * atomic_mass  # kg
unit_time = sigma * (mass / epsilon) ** 0.5  # s
unit_pressure = epsilon/sigma**3  # Pa
unit_density = mass/sigma**3  # kg/m^3
cf = dict(
    sigma=sigma,  # m
    in_Angstrom=sigma_in_Angstrom,
    epsilon=epsilon,   # J
    in_K=epsilon_in_K,
    in_kJ_per_mol=epsilon*1e-3*Avogadro,
    mass=mass,  # kg
    in_u=mass_in_u,
    unit_time=unit_time,
    in_ns=unit_time*1e9,
    in_ps=unit_time*1e12,
    in_fs=unit_time*1e15,
    unit_pressure=unit_pressure,  # Pa
    in_bar=unit_pressure*1e-5,
    in_MPa=unit_pressure*1e-6,
    in_GPa=unit_pressure*1e-9,
    unit_density=unit_density,  # kg/m^3
    in_g_per_ml = unit_density*1e-3
)


# Examples of how a general function doing this could work:
# cf = gp.conversion_factors(sigma_in_m = 3.4e-10, epsilon_in_K = 120, mass_in_u = 39.948)  # Argon units
# cf = gp.conversion_factors(sigma_in_Angstrom= 3.4e-10, epsilon_in_kJ_per_mol = 3.23, mass_in_kg = 39.948e32)
# cf =  gp.conversion_factors(sigma_in_m = 3.4e-10, epsilon_in_J = ..., mass_in_kg = ...)) # Do simulation in SI units (no advisable!)
# cf =  gp.conversion_factors(sigma_in_Angstrom = 1, epsilon_in_kJ_per_mol = 1,  mass_in_u = 1)) # Do simulation in molar units units (no advisable!)

print('Dictionary with conversion factors (from reduced units to SI units):')
pprint(cf)

# Set up simulation in reduced units
configuration = gp.Configuration(D=3)
density_in_g_per_ml = 1.6  # g/ml (aka g/cm^3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=density_in_g_per_ml/cf['in_g_per_ml'])
configuration['m'] = 39.948/cf['in_u']  # 1.0 in reduced units
temperature_in_K = 84  # K
configuration.randomize_velocities(temperature=temperature_in_K/cf['in_K'])
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5  # in reduced units
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
dt_in_fs = 5.0  # fs
tau_in_ps = 0.5  # ps
integrator = gp.integrators.NVT(temperature=temperature_in_K/cf['in_K'], tau=tau_in_ps/cf['in_ps'], dt=dt_in_fs/cf['in_fs'])
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]
sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=8, steps_per_timeblock=1*1024,
                    storage='memory')

# Run simulation and print information in SI units
for timeblock in sim.run_timeblocks():
    time = timeblock * sim.steps_per_block * sim.dt
    time_in_ps = timeblock * sim.steps_per_block * sim.dt * cf['in_ps']
    energy =  np.mean(configuration['U'])
    energy_in_kJ_per_mol = np.mean(configuration['U']) * cf['in_kJ_per_mol']
    N, D = configuration.N, configuration.D
    dof = D * N - D
    T_kin = 2.0 * np.sum(configuration['K']) / dof
    T_kin_in_K = T_kin * cf['in_K']
    volume = configuration.get_volume()
    rho = N/volume
    P = rho * T_kin + np.sum(configuration['W']) / volume
    pressure_in_MPa = P * cf['in_MPa']
    print(f't = {time_in_ps:.1f} ps,  u = {energy_in_kJ_per_mol:.2f} kJ/mol,  T_kin = {T_kin_in_K:.1f} K,  P = {pressure_in_MPa:.1f} MPa (t = {time:.1f},  u = {energy:.2f},  T_kin = {T_kin:.1f},  P = {P:.1f})')
