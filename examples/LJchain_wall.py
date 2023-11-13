import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt
import math


include_springs = False

wall_dimension = 2 # Could use normal vector to be even more general...
nxy, nz = 8, 4
N = nxy*nxy*nz*4 # FCC

rho = 0.85
Lz = 6.13
Lxy = (N/Lz/rho)**0.5
print('Density:', N/Lxy/Lxy/Lz )
# Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density 
positions, simbox_data = rp.generate_fcc_positions(nx=nxy, ny=nxy, nz=nz, rho=1.5)
print(simbox_data)
simbox_data[:] = Lxy
simbox_data[wall_dimension] = Lz # Make room for walls
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

# Make smooth wall
wall_indicies = np.zeros((N, 3), dtype=np.int32)
wall_indicies[:,0] = np.arange(N)    # All particles feel the wall
wall_indicies[:,1] = 0               # Only one wall type
wall_indicies[:,2] = wall_dimension  # plane of wall 
wall_params = np.zeros((1, D+3), dtype=np.float32)
wall_params[0,2] =  simbox_data[wall_dimension]/2.0 
print('Wall at +-', wall_params[0,2])
prefactor = 4.0*math.pi/3*rho      # [Ingebrigtsen, Dyre, Soft Matter, 2014]
wall_params[0,3] =  prefactor/15.0 # Am
wall_params[0,4] = -prefactor/2.0  # An
wall_params[0,5] =  2.5 # cutoff
wall_function = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(9,3))
wall_calculator = rp.make_smooth_wall_calculator(c1, wall_function)
wall_interactions = rp.make_fixed_interactions(c1, wall_calculator, compute_plan, verbose=True)
d_wall_indicies = cuda.to_device(wall_indicies)
d_wall_params = cuda.to_device(wall_params)
wall_interaction_params = (d_wall_indicies, d_wall_params)

# Make bond interactions
if include_springs:
    bond_indicies = np.zeros((N//2, 3), dtype=np.int32)
    even = np.arange(0,N,2)
    bond_indicies[:,0] = even
    bond_indicies[:,1] = even+1
    bond_params = np.zeros((N//2, 2), dtype=np.float32)
    bond_params[:,0] = 2**(1/6)
    bond_params[:,1] = 57.15 
    bond_calculator = rp.make_bond_calculator(c1, rp.harmonic_bond_function)
    bond_interactions = rp.make_fixed_interactions(c1, bond_calculator, compute_plan, verbose=True)
    d_bond_indicies = cuda.to_device(bond_indicies)
    d_bond_params = cuda.to_device(bond_params)
    bond_interaction_params = (d_bond_indicies, d_bond_params)

# Prepare exlusions from bonds
exclusions = np.zeros((N,10), dtype=np.int32)
if include_springs:
    rp.add_exclusions_from_bond_indicies(exclusions, bond_indicies)
d_exclusions = cuda.to_device(exclusions)

# Make the pair interactions
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

# Add up interactions
if include_springs:
    interactions = rp.add_interactions_list(c1, (pair_interactions, bond_interactions, wall_interactions), compute_plan, verbose=True,)
    interaction_params = (pair_interaction_params, bond_interaction_params, wall_interaction_params)
else:
    interactions = rp.add_interactions_list(c1, (pair_interactions, wall_interactions), compute_plan, verbose=True,)
    interaction_params = (pair_interaction_params, wall_interaction_params)

    
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

scalars_t = []
coordinates_t = []
tt = []

#inner_steps = 1000
#steps = 500
equil_steps = 500000
inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()

#Equilibration
integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, equil_steps)

start.record()
for i in range(steps):
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, inner_steps)
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
    c1.copy_to_host()
    coordinates_t.append(c1['r'][:,wall_dimension])
    tt.append(i*inner_steps*dt)

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
    
rp.plot_scalars(df, N, D, figsize=(15,4))

if include_springs:
    c1.copy_to_host()
    lengths = []

    for i in range(N//2):
        dist_sq = c1.simbox.dist_sq_function(c1['r'][bond_indicies[i,0]], c1['r'][bond_indicies[i,1]], c1.simbox.data)
        dist = math.sqrt(dist_sq)
        if dist>1.5:
            print(i, bond_indicies[i,0], bond_indicies[i,1], dist)
        lengths.append(dist)

    plt.hist(lengths, bins=30)
    plt.show()

bins = 300
hist, bin_edges = np.histogram(np.array(coordinates_t).flatten(), bins=bins, range=(-Lz/2, +Lz/2))
dx = bin_edges[1] - bin_edges[0]
x = bin_edges[:-1]+dx/2 
y = hist/len(coordinates_t)/Lxy**2/dx
plt.plot(x, y)
plt.show()




