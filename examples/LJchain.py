import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt
import math

# Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density 
positions, simbox_data = rp.generate_fcc_positions(nx=4, ny=8, nz=8, rho=0.8442)
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

# Make bond interactions
bond_indicies = np.zeros((N//2, 2), dtype=np.int32)
even = np.arange(0,N,2)
bond_indicies[:,0] = even
bond_indicies[:,1] = even+1
bond_params = np.zeros((N//2, 2), dtype=np.float32)
bond_params[:,0] = 0.9
bond_params[:,1] = 1000.0 
bond_calculator = rp.make_bond_calculator(c1, rp.harmonic_bond_function)
bond_interactions = rp.make_fixed_interactions(c1, bond_calculator, compute_plan, verbose=True)
d_bond_indicies = cuda.to_device(bond_indicies)
d_bond_params = cuda.to_device(bond_params)
bond_interaction_params = (d_bond_indicies, d_bond_params)

# Prepare exlusions from bonds
exclusions = np.zeros((N,10), dtype=np.int32)
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
interactions = rp.add_interactions(c1, pair_interactions, bond_interactions, compute_plan, verbose=True,)
interaction_params = (pair_interaction_params, bond_interaction_params)
    
# Make integrator
integrator_step = rp.make_step_nve(c1, compute_plan=compute_plan, verbose=True)
integrate = rp.make_integrator(c1, integrator_step, interactions, compute_plan=compute_plan, verbose=True)
dt = np.float32(0.0025)
integrator_params = (dt, )

scalars_t = []
tt = []

#inner_steps = 1000
#steps = 500
inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()

for i in range(steps+1):
    if i==1:
        start.record() # Exclude first step from timing to get it more precise by excluding time for JIT compiling
   
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data, interaction_params, integrator_params, inner_steps)
    scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
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
if compute_plan['UtilizeNIII']: # This correction should not be necesarry in user-land
    df['u'] *= 2
    df['w'] *= 2
    df['lap'] *= 2
df['w'] *= 1/D/2
df['t'] = np.array(tt)  
    
rp.plot_scalars(df, N, D, figsize=(15,4))

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


