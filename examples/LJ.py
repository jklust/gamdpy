import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd

# Generate numpy arrays for particle positions and simbox of a FCC lattice with a given density 
positions, simbox_data = rp.generate_fcc_positions(nx=4, ny=8, nz=8, rho=0.8442) # 1024
#positions, simbox_data = rp.generate_fcc_positions(nx=16, ny=16, nz=16, rho=0.8442) # 16*1024
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

# Should not be necessarry:
exclusions = np.zeros((N,10), dtype=np.int32)
d_exclusions = cuda.to_device(exclusions)

# Make pair potential
pair_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(12,6))
params = [[[4.0, -4.0, 2.5],], ]
#pair_potential = rp.apply_shifted_force_cutoff(rp.make_IPL_n(10))
#params = [[[1.0, 2.5],], ]
LJ = rp.PairPotential(c1, pair_potential, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = LJ.get_interactions(c1, exclusions=d_exclusions, compute_plan=compute_plan, verbose=True)

# Make integrator
integrator_step = rp.make_step_nve(c1, compute_plan=compute_plan, verbose=True)
integrate = rp.make_integrator(c1, integrator_step, pairs['interactions'], compute_plan=compute_plan, verbose=True)

dt = np.float32(0.005)
integrator_params = (dt, )

scalars_t = []
tt = []

inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()

for i in range(steps+1):
    if i==1:
        start.record() # Exclude first step from timing to get it more precise by excluding time for JIT compiling
        
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params, inner_steps)
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
df['t'] = np.array(tt)  
    
rp.plot_scalars(df, N, D, figsize=(15,4))


