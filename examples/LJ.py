import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd

# Generate configuration with a FCC lattice
c1 = rp.make_configuration_fcc(nx=4,  ny=8,  nz=8,  rho=0.8442, T=1.44) 
c1.copy_to_device() 

compute_plan = rp.get_default_compute_plan(c1)
print('compute_plan: ', compute_plan)

# Make pair potential
pair_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(12,6))
params = [[[4.0, -4.0, 2.5],], ]
LJ = rp.PairPotential(c1, pair_potential, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = LJ.get_interactions(c1, exclusions=None, compute_plan=compute_plan, verbose=True)

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
zero = np.float32(0.0)

for i in range(steps+1):
    if i==1:
        start.record() # Exclude first step from timing to get it more precise by excluding time for JIT compiling
        
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params, zero, inner_steps)
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
rp.plot_scalars(df, c1.N, c1.D, figsize=(15,4))


