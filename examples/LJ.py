import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd
import matplotlib.pyplot as plt

include_rdf = True

# Generate configuration with a FCC lattice
c1 = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=0.8442, T=1.44)      # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=16,  ny=16,  nz=16,  rho=0.8442, T=1.44)  # N = 16*1024
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

# Make rdf calculator
if include_rdf:
    num_bins = 500
    full_range = True
    gr_bins = np.zeros(num_bins, dtype=np.float64)
    d_gr_bins = cuda.to_device(gr_bins)
    host_array_zeros = np.zeros(d_gr_bins.shape, dtype=d_gr_bins.dtype)
    rdf_calculator = rp.make_rdf_calculator(c1, pair_potential = LJ, compute_plan=compute_plan, full_range = full_range, verbose=True)  

#  Run Simulation 
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

    if include_rdf:
        rdf_calculator(c1.d_vectors, c1.simbox.d_data, c1.d_ptype, pairs['interaction_params'], d_gr_bins)
        temp_host_array = d_gr_bins.copy_to_host()       # offloading data from device and resetting decive array to zero. 
        gr_bins += temp_host_array                       # ... (prevents overflow errors for longer runs)  
        d_gr_bins = cuda.to_device(host_array_zeros)

end.record()
end.synchronize()
timing_numba = cuda.event_elapsed_time(start, end)
nbflag = LJ.nblist.d_nbflag.copy_to_host()    
tps = steps*inner_steps/timing_numba*1000

print('\tsteps :', steps*inner_steps)
print('\tnbflag : ', nbflag)
print('\ttime :', timing_numba/1000, 's')
print('\tTPS : ', tps )

if include_rdf:
    data = rp.normalize_and_save_gr(gr_bins, c1, pairs['interaction_params'], full_range, steps)
    plt.figure()
    plt.plot(data[:,0], data[:,1], '-')
    plt.xlabel('distance')
    plt.ylabel('Radial distribution function')
    plt.show(block=False)

df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
df['t'] = np.array(tt)      
rp.plot_scalars(df, c1.N, c1.D, figsize=(15,4))


