import numpy as np
import rumdpy as rp

rp.select_gpu()

# Simulation params 
rho, temperature = 0.85, 1.5
N_A, N_B, N_C = 8, 4, 4  # Number of atoms of each tyoe
particles_per_molecule = N_A + N_B + N_C
filename = 'Data/molecules'
num_timeblocks = 64
steps_per_timeblock = 1 * 1024 # 8 * 1024 to show reliable pattern formation

positions = []
particle_types = []
masses = []

# A particles
for i in range(N_A):
    positions.append( [ i*1.0, (i%2)*.1, 0. ] ) # x, y, z for this particle
    particle_types.append( 0 )
    masses.append( 1.0 )  

# B particles
for i in range(N_B):
    positions.append( [ 0, (i+1)*1.0, ((i+1)%2)*.1 ] ) # x, y, z for this particle
    particle_types.append( 1 )
    masses.append( 1.0 )  

# C particles
for i in range(N_C):
    positions.append( [ ((i+1)%2)*.1, 0, (i+1)*1.0 ] ) # x, y, z for this particle
    particle_types.append( 2 )
    masses.append( 1.0 )  

# Setup configuration: Single molecule first, then duplicate
top = rp.Topology(['MyMolecule', ])
top.bonds = rp.bonds_from_positions(positions, cut_off=1.1, bond_type=0)
top.angles = rp.angles_from_bonds(top.bonds, angle_type=0)
top.dihedrals = rp.dihedrals_from_angles(top.angles, dihedral_type=0)
top.molecules['MyMolecule'] = rp.molecules_from_bonds(top.bonds)

print('Initial Positions:')
for position in positions:
    print('\t\t', position)
print('Particle types:\t', particle_types)
print('Bonds:         \t', top.bonds)
print('Angles:        \t', top.angles)
print('Dihedrals:     \t', top.dihedrals)
print()

# This call creates the pdf "molecule.pdf" with a drawing of the molecule 
# Use block=True to visualize the molecule before running the simulation
rp.plot_molecule(top, positions, particle_types, filename="molecule.pdf", block=False)

#configuration = rp.duplicate_molecule(top, positions, particle_types, masses, cells=(6, 6, 6), safety_distance=2.0)
configuration = rp.replicate_molecules([top], [positions], [particle_types], [masses], [216], safety_distance=2.0, compute_flags={"stresses":True})


configuration.randomize_velocities(temperature=temperature)

print(f'Number of molecules: {len(configuration.topology.molecules["MyMolecule"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = rp.harmonic_bond_function
bond_params = [[0.8, 1000.], ]
bonds = rp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Make angle interactions
angle_potential = rp.cos_angle_function
angle0, k = 2.0, 500.0
angles = rp.Angles(angle_potential, configuration.topology.angles, parameters=[[k, angle0],]) 

# Make dihedral interactions
dihedral_potential = rp.ryckbell_dihedral
rbcoef=[.0, 5.0, .0, .0, .0, .0]    
dihedrals = rp.Dihedrals(dihedral_potential, configuration.topology.dihedrals, parameters=[rbcoef, ])

# Exlusion list
exclusions = dihedrals.get_exclusions(configuration)
#exclusions = bonds.get_exclusions(configuration)

# Make pair potential
pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 1.00, 1.00],
       [1.00, 1.00, 1.00],
       [1.00, 1.00, 1.00],]
eps = [[1.00, 1.00, 1.00],
       [1.00, 1.00, 0.80],
       [1.00, 0.80, 1.00],]
cut = [[2.50, 1.12, 1.12],
       [1.12, 2.50, 2.50],
       [1.12, 2.50, 2.50]]

pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
integrator = rp.integrators.NVT(temperature=temperature, tau=0.1, dt=0.004)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [rp.ConfigurationSaver(), 
                   rp.ScalarSaver(32),
                   rp.StressSaver(32, compute_flags={'stresses':True}),
                   rp.MomentumReset(100)]

# Setup simulation
sim = rp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

print('\nCompression and equilibration: ')
dump_filename = 'Data/dump_compress.lammps'
with open(dump_filename, 'w') as f:
    print(rp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    print(sim.status(per_particle=True), f'rho= {N/volume:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    with open(dump_filename, 'a') as f:
        print(rp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks/2:
        desired_rho = (block+1)/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)

sim = rp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
dump_filename = 'Data/dump.lammps'
with open(dump_filename, 'w') as f:
    print(rp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(rp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary()) 
print(configuration)

W = rp.extract_scalars(sim.output, 'W')
full_stress_tensor = rp.extract_stress_tensor(sim.output)
mean_diagonal_sts = (full_stress_tensor[:,0,0] + full_stress_tensor[:,1,1] + full_stress_tensor[:,2,2])/3

print("Mean diagonal stress", np.mean(mean_diagonal_sts) )
print("Pressure", np.mean(W)*rho/N + temperature*rho)

print('\nAnalyse structure with:')
print('   python3 analyze_structure.py Data/molecules')

print('\nAnalyze dynamics with:')
print('   python3 analyze_dynamics.py Data/molecules')

print('\nVisualize simulation in ovito with:')
print('   ovito Data/dump.lammps')

#print('\nVisualize simulation in VMD with:')
#print('   vmd -lammpstrj Data/dump.lammps')
