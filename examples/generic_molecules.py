

import numpy as np
import rumdpy as rp

# Sim. params 
rho, temperature = 1.0, 1.5
NVE = False  # If True -> k small
angle0, k = 2.0, 500.0
#rbcoef=[15.5000,  20.3050, -21.9170, -5.1150,  43.8340, -52.6070]
rbcoef=[.0, 50.0, .0, .0, .0, .0]

# Generate configuration with a FCC lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=(8, 8, 8), rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=temperature)

# Make bonds
bond_potential = rp.harmonic_bond_function
bond_params = [[1.0, 1000.], ]
bond_indices = []
for n in range(0, configuration.N, 4):
    bond_indices.append([n, n+1, 0])
    bond_indices.append([n+1, n+2, 0])
    bond_indices.append([n+2, n+3, 0])

bonds = rp.Bonds(bond_potential, bond_params, bond_indices)

# Angles
angle_params = [[k, angle0],]
angle_indices = []
for n in range(0, configuration.N, 4):
    angle_indices.append([n, n+1, n+2, 0])
    angle_indices.append([n+1, n+2, n+3, 0])

angles = rp.Angels(angle_indices, angle_params) 

# Dihedrals
dihedral_params = [rbcoef, ]
dihedral_indices = []
for n in range(0, configuration.N, 4):
    dihedral_indices.append([n, n+1, n+2, n+3, 0])

dihedrals = rp.Dihedrals(dihedral_indices, dihedral_params)

# Exlusion list
#exclusions = angles.get_exclusions(configuration)
#exclusions = bonds.get_exclusions(configuration)
exclusions = dihedrals.get_exclusions(configuration)


# Make pair potential
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
if NVE:
    integrator = rp.integrators.NVE(dt=0.001)
else:
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.1, dt=0.002)

# Compute plan
compute_plan = rp.get_default_compute_plan(configuration)

# Setup simulation
sim = rp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator,
                    num_timeblocks=10, steps_per_timeblock=256,
                    steps_between_momentum_reset=100,
                    compute_plan=compute_plan, storage='memory')

angles_array, dihedrals_array = [], []
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))     
    angles_array.append( angles.get_angle(10, configuration) )
    dihedrals_array.append( dihedrals.get_dihedral(10, configuration) )

print(sim.summary()) 

columns = ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol'] 
data = np.array(rp.extract_scalars(sim.output, columns, first_block=1)) 
temp = 2.0/3.0*np.mean(data[4])/configuration.N
Etot = data[0] + data[4]
Etot_mean = np.mean(Etot)/configuration.N
Etot_std = np.std(Etot)/configuration.N

print("Temp:  %.2f  Etot: %.2e (%.2e)" % (temp,  Etot_mean, Etot_std))
print("Angle: %.2f (%.2f) " % (np.mean(angles_array), np.std(angles_array)))
print("Dihedral: %.2f (%.2f) " % (np.mean(dihedrals_array), np.std(dihedrals_array)))

