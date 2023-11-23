import numpy as np
import rumdpy as rp
import numba
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt
import math

include_springs = True
include_walls = True

wall_dimension = 2
nxy, nz = 8, 4
N = nxy*nxy*nz*4 # FCC

rho = 0.85
wall_dist = 6.31
Lxy = (N/wall_dist/rho)**0.5
print('Density:', N/Lxy/Lxy/wall_dist )

# Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density 
positions, simbox_data = rp.generate_fcc_positions(nx=nxy, ny=nxy, nz=nz, rho=1.5)
print(simbox_data)
simbox_data[:] = Lxy
simbox_data[wall_dimension] = wall_dist+2 # 
print(simbox_data)
N, D = positions.shape

### Make configuration. Could be read from file or generated from single convenience function, but this shows flexibility
c1 = rp.Configuration(N, D, simbox_data)
c1['r'] = positions
c1['v'] = rp.generate_random_velocities(N, D, T=1.44)
c1['m'] =  np.ones(N, dtype=np.float32)     # Set masses
c1.ptype = np.zeros(N, dtype=np.int32)      # Set types
c1.copy_to_device()           

compute_plan = rp.get_default_compute_plan(c1)
print('compute_plan: ', compute_plan)
 
# Make bond interactions (This is the bare-bones way - It should be possible to setup and replicate molecules)
if include_springs:
    bond_potential = rp.harmonic_bond_function
    potential_params_list = [[1.12, 1000.], [1.0, 1000.], [1.12, 1000.]]
    fourth = np.arange(0,N,4)
    bond_particles_list = [np.array((fourth, fourth+1)).T, np.array((fourth+1, fourth+2)).T, np.array((fourth+2, fourth+3)).T] 
    bond_interactions, bond_interaction_params = rp.setup_bonds(c1, bond_potential, potential_params_list, bond_particles_list, compute_plan, verbose=True)
    
# Prepare exclusions from bonds (Should be more automatic)
exclusions = np.zeros((N,10), dtype=np.int32)
if include_springs:
    for bond_particles in bond_particles_list:
        rp.add_exclusions_from_bond_indicies(exclusions, bond_particles)
d_exclusions = cuda.to_device(exclusions)
    
# Make two smooth walls
if include_walls:
    wall_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(9,3))
    prefactor = 4.0*math.pi/3*rho
    potential_params_list = [[prefactor/15.0, -prefactor/2.0, 3.0], [prefactor/15.0, -prefactor/2.0, 3.0]] # Ingebrigtsen & Dyre (2014)
    particles_list =        [np.arange(N),                           np.arange(N)]                         # All particles feel the wall(s)
    wall_point_list =       [[0, 0, wall_dist/2.0],                  [0, 0, -wall_dist/2.0] ]
    normal_vector_list =    [[0,0,1],                                [0,0,1]]
    wall_interactions, wall_interaction_params = rp.setup_planar_interactions(c1, wall_potential, potential_params_list, 
                                                                       particles_list, wall_point_list, normal_vector_list, compute_plan, verbose=True)
    
# Make pair interactions
cut_off = 2.5
params = np.zeros((1,1), dtype="f,f,f")
params[0][0] = (4., -4., cut_off)
print('Pairpotential paramaters:\n', params)
skin = np.float32(compute_plan['skin'])
max_cut = np.float32(cut_off)
LJ = rp.PairPotential(c1, rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(12,6)), params=params, max_num_nbs=1000, compute_plan=compute_plan)
num_cscalars = 3
pair_interactions = rp.make_interactions(c1, pair_potential = LJ, num_cscalars=num_cscalars, compute_plan=compute_plan, verbose=True,)
LJ.copy_to_device()
pair_interaction_params = (LJ.d_params, max_cut, skin, LJ.nblist.d_nblist,  LJ.nblist.d_nbflag, d_exclusions)


# Add up interactions (Needs to start with pair-interactions, and there can only be one...)
interactions_list = [pair_interactions,]
interaction_params_list = [pair_interaction_params, ]

if include_walls:
    interactions_list.append(wall_interactions)
    interaction_params_list.append(wall_interaction_params)

if include_springs:
    interactions_list.append(bond_interactions)
    interaction_params_list.append(bond_interaction_params)

interactions, interaction_params = rp.add_interactions_list(c1, interactions_list, interaction_params_list, compute_plan, verbose=True,)
    
# Make NVT integrator
integrator_step = rp.make_step_nvt(c1, compute_plan=compute_plan, verbose=True)
integrate = rp.make_integrator(c1, integrator_step, interactions, compute_plan=compute_plan, verbose=True)
dt = np.float32(0.0025)
T = np.float32(1.8)
tau=0.2
omega2 = np.float32(4.0*np.pi*np.pi/tau/tau)
degrees = N*D - D
thermostat_state = np.zeros(2, dtype=np.float32)
d_thermostat_state = cuda.to_device(thermostat_state)
integrator_params =  (dt, T, omega2, degrees,  d_thermostat_state)
integrator_params_initial =  (dt, np.float32(10.0), omega2, degrees,  d_thermostat_state)

# move dt and T to integrator call? Performance cost?

scalars_t = []
coordinates_t = []
tt = []

#inner_steps = 1000
#steps = 500q
equil_steps = 30000
inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()

dr = np.zeros(3)
dz = np.array((0., 0., 1.))

@numba.njit()
def get_bond_lengths_theta_z(r, bond_indicies, dist_sq_dr_function, simbox_data):
    bond_lengths = np.zeros(bond_indicies.shape[0], dtype=np.float32)
    theta_z = np.zeros(bond_indicies.shape[0])
    dr = np.zeros(3)

    for j in range(bond_indicies.shape[0]):
        dist_sq = dist_sq_dr_function(r[bond_indicies[j,0]], r[bond_indicies[j,1]], simbox_data, dr)
        dist = math.sqrt(dist_sq)
        bond_lengths[j] = dist
        theta_z[j] = math.acos(abs(dr[2]/dist))/math.pi*180
    return bond_lengths, theta_z

#Equilibration
integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params_initial, equil_steps)
integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, equil_steps)
bond_lengths = []
theta_z = []

f = numba.njit(c1.simbox.dist_sq_dr_function)

start.record()
for i in range(steps):
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, inner_steps)
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
    tt.append(i*inner_steps*dt)

    c1.copy_to_host()
    coordinates_t.append(c1['r'][:,wall_dimension])
    if include_springs:
        for bond_particles in bond_particles_list:
            lengths, theta = get_bond_lengths_theta_z(c1['r'], bond_particles, f, c1.simbox.data)
            bond_lengths.append(lengths)
            theta_z.append(theta)

end.record()
end.synchronize()
timing_numba = cuda.event_elapsed_time(start, end)
nbflag = LJ.nblist.d_nbflag.copy_to_host()    
tps = steps*inner_steps/timing_numba*1000

print('\tsteps :', steps*inner_steps)
print('\tnbflag : ', nbflag)
print('\ttime :', timing_numba/1000, 's')
print('\tTPS : ', tps )
   
df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
df['t'] = np.array(tt)  
    
rp.plot_scalars(df, N, D, figsize=(15,4), block=False)

if include_springs:
    plt.figure() 
    plt.hist(np.array(bond_lengths).flatten(), bins=100, density=True)
    plt.xlabel('bond length')
    plt.ylabel('p(bond length)')
    plt.show(block=False)
    
    # what is the distribution of theta_z for random directions
    dr = np.random.randn(steps*N, D)
    dl = np.sum(dr*dr, axis=1)**0.5
    dr /= np.tile(dl,(3,1)).T
    theta_z_random = np.arccos(np.abs(dr[:,0]))/math.pi*180
    
    plt.figure() 
    bins = 100
    hist, bin_edges = np.histogram(theta_z_random, bins=bins, range=(0, 90), density=True)
    dx = bin_edges[1] - bin_edges[0]
    x = bin_edges[:-1]+dx/2 
    plt.plot(x,hist, label='Random')

    hist, bin_edges = np.histogram(np.array(theta_z).flatten(), bins=bins, range=(0, 90), density=True)
    plt.plot(x,hist, label='Simulation')
    plt.xlabel('Theta (angle with z-axis)')
    plt.ylabel('p(Theta)')
    plt.legend()
    plt.show(block=False)

    
bins = 300
hist, bin_edges = np.histogram(np.array(coordinates_t).flatten(), bins=bins, range=(-wall_dist/2, +wall_dist/2))
dx = bin_edges[1] - bin_edges[0]
x = bin_edges[:-1]+dx/2 
y = hist/len(coordinates_t)/Lxy**2/dx
plt.figure()
plt.plot(x, y)
plt.plot(x, np.ones_like(x)*rho, '--', label=f'rho={rho}')
plt.xlabel('z')
plt.ylabel('rho(z)')
plt.show()

