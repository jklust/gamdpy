import numpy as np
import rumdpy as rp
import matplotlib.pyplot as plt

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

configuration = rp.duplicate_molecule(top, positions, particle_types, masses, cells=(6, 6, 6), safety_distance=2.0)
configuration.randomize_velocities(temperature=temperature)

print(f'Number of molecules: {len(configuration.topology.molecules["MyMolecule"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = rp.harmonic_bond_function
bond_params = [[0.8, 1000.], ]
bonds = rp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Make angle interactions
angle0, k = 2.0, 500.0
angles = rp.Angles(configuration.topology.angles, parameters=[[k, angle0],]) 

# Make dihedral interactions
rbcoef=[.0, 5.0, .0, .0, .0, .0]    
dihedrals = rp.Dihedrals(configuration.topology.dihedrals, parameters=[rbcoef, ])

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
                   rp.ScalarSaver(), 
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
# Setup on-the-fly calculation of Radial Distribution Function
calc_rdf = rp.CalculatorRadialDistribution(configuration, bins=300)

print('\nProduction: ')
dump_filename = 'Data/dump.lammps'
with open(dump_filename, 'w') as f:
    print(rp.configuration_to_lammps(sim.configuration, timestep=0), file=f)

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(rp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    if block > sim.num_blocks//2:
        calc_rdf.update()
print(sim.summary()) 
print(configuration)

print('Open dump file in ovito with:')
print('   ovito Data/dump.lammps')

print('Open in VMD with:')
print('   vmd -lammpstrj Data/dump.lammps')


dynamics = rp.tools.calc_dynamics(sim.output, first_block=sim.num_blocks//2)
plt.figure()
num_type = dynamics['msd'].shape[1]
plt.loglog(dynamics['times'], dynamics['msd'][:,0], 'o-', label='A')
if num_type>1: 
    plt.loglog(dynamics['times'], dynamics['msd'][:,1], 'o-', label='B') 
if num_type>2:
    plt.loglog(dynamics['times'], dynamics['msd'][:,2], 'o-', label='C')
factor = np.array([1, 10])
plt.loglog(dynamics['times'][0]*factor, dynamics['msd'][0,num_type-1]*factor**2, 'k--', alpha=0.5)
plt.loglog(dynamics['times'][-1]/factor, dynamics['msd'][-1,num_type-1]/factor, 'k--', alpha=0.5)
plt.legend()
plt.savefig(filename+"_msd.pdf", format="pdf", bbox_inches="tight")
plt.show(block=False)

rdf = calc_rdf.read()
plt.figure()
rdf_data = np.mean(rdf['rdf_ptype'], axis=0)
num_type = rdf_data.shape[0]
plt.plot(rdf['distances'], rdf_data[0,0,:], '-', label='A-A')
if num_type>1:
    plt.plot(rdf['distances'], rdf_data[1,0,:], '-', label='B-A')
    plt.plot(rdf['distances'], rdf_data[1,1,:], '-', label='B-B')
if num_type>2:
    plt.plot(rdf['distances'], rdf_data[2,0,:], '-', label='C-A')
    plt.plot(rdf['distances'], rdf_data[2,1,:], '-', label='C-B')
    plt.plot(rdf['distances'], rdf_data[2,2,:], '-', label='C-C')

plt.legend()
plt.savefig(filename+"_rdf.pdf", format="pdf", bbox_inches="tight")
plt.show(block=True)

   
columns = ['U', 'W', 'K',] 
data = np.array(rp.extract_scalars(sim.output, columns, first_block=1))
temp = 2.0/3.0*np.mean(data[2])/configuration.N
Etot = data[0] + data[2]
Etot_mean = np.mean(Etot)/configuration.N
Etot_std = np.std(Etot)/configuration.N

print('\nFinal molecular potential energies: ')
molecular_energies = np.array( [ np.sum(configuration['U'][atoms])
                      for atoms in configuration.topology.molecules['MyMolecule'] ])
print(molecular_energies[:15])

print(np.mean(molecular_energies), np.mean(configuration['U'])*particles_per_molecule)
assert np.isclose(np.mean(molecular_energies), np.mean(configuration['U'])*particles_per_molecule)
