import numpy as np
import rumdpy as rp
from rumdpy.integrators import nve, nve_toxvaerd, nvt, nvt_langevin, npt_langevin
from numba import cuda
import pandas as pd
import pickle
import json
import sys
#import matplotlib.pyplot as plt

include_rdf = True
if 'NoRDF' in sys.argv:
    include_rdf = False
    
integrator = 'NVE'
if 'NVE_Toxvaerd' in sys.argv:
    integrator = 'NVE_Toxvaerd'
if 'NVT' in sys.argv:
    integrator = 'NVT'
if 'NVT_Langevin' in sys.argv:
    integrator = 'NVT_Langevin'
if 'NPT_Langevin' in sys.argv:            # use with NoRDF since box size is varying
    integrator = 'NPT_Langevin'


# Generate configuration with a FCC lattice
c1 = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=0.8442,  T=1.44)  # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=1./0.9672,  T=1.44)  # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=16,  ny=16,  nz=16,  rho=0.8442, T=1.44)  # N = 16*1024
c1.copy_to_device() 

pdict = {'N':c1.N, 'D':c1.D, 'simbox':c1.simbox.data, 'integrator':integrator, 'rdf':include_rdf}
print(pdict)
with open('Data/LJ_pdict.pkl', 'wb') as f:
    pickle.dump(pdict, f)

compute_plan = rp.get_default_compute_plan(c1)

# Make pair potential
pair_potential = rp.apply_shifted_force_cutoff(rp.make_LJ_m_n(12,6))
params = [[[4.0, -4.0, 2.5],], ]
LJ = rp.PairPotential(c1, pair_potential, params=params, max_num_nbs=1000, compute_plan=compute_plan)
pairs = LJ.get_interactions(c1, exclusions=None, compute_plan=compute_plan, verbose=True)

# Make integrator
dt = 0.005
T0 = rp.make_function_constant(value=0.7) # Not used for NVE
#T0 = rp.make_function_constant(value=1.20) # Not used for NVE
P0 = rp.make_function_constant(value=1.2) # Not used for NV*
#P0 = rp.make_function_constant(value=2.2) # Not used for NV*

if integrator=='NVE':
    integrate, integrator_params = nve.setup(c1, pairs['interactions'], dt=dt, compute_plan=compute_plan, verbose=False)

if integrator=='NVE_Toxvaerd':
    integrate, integrator_params = nve_toxvaerd.setup(c1, pairs['interactions'], dt=dt, compute_plan=compute_plan, verbose=False)

if integrator=='NVT':
    integrate, integrator_params =nvt.setup(c1, pairs['interactions'], T0, tau=0.2, dt=dt, compute_plan=compute_plan, verbose=False) 
        
if integrator=='NVT_Langevin':
    integrate, integrator_params = nvt_langevin.setup(c1, pairs['interactions'], T0, alpha=0.1, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

if integrator=='NPT_Langevin':
    integrate, integrator_params = npt_langevin.setup(c1, pairs['interactions'], T0, P0, alpha=0.1, alpha_baro=0.0001, 
        mass_baro=0.0001, volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

# Make rdf calculator
if include_rdf:
    num_bins = 500
    full_range = True
    gr_bins = np.zeros(num_bins, dtype=np.float64)
    d_gr_bins = cuda.to_device(gr_bins)
    host_array_zeros = np.zeros(d_gr_bins.shape, dtype=d_gr_bins.dtype)
    rdf_calculator = rp.make_rdf_calculator(c1, pair_potential = LJ, compute_plan=compute_plan, 
                                            full_range=full_range, verbose=True)  

#  Run Simulation 
scalars_t = []
tt = []
vol_t = []

inner_steps = 1000
steps = 500

start = cuda.event()
end = cuda.event()
zero = np.float32(0.0)

for i in range(steps+1):
    if i==1:
        start.record() # Exclude first step from timing to get it more precise by excluding time for JIT compiling
        
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params, zero, inner_steps)
    
    if i>0:
        scalars_t.append(np.sum(c1.d_scalars.copy_to_host(), axis=0))
        tt.append(i*inner_steps*dt)        
        c1.simbox.data = c1.simbox.d_data.copy_to_host()
        vol = (c1.simbox.data[0] * c1.simbox.data[1] * c1.simbox.data[2])
        vol_t.append(vol)

    if i>0 and include_rdf:
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
print('\tfinal box dims : ', c1.simbox.data[0], c1.simbox.data[1], c1.simbox.data[2])

# Save data
df = pd.DataFrame(np.array(scalars_t), columns=c1.sid.keys())
df['t'] = np.array(tt)
df['vol'] = np.array(vol_t)
print(df.mean())
df.to_csv('Data/LJ_scalars.csv')

if include_rdf:
    data = rp.normalize_and_save_gr(gr_bins, c1, pairs['interaction_params'], 
                                    full_range, steps, filename='Data/LJ_rdf.dat')

# Plot results
import analyze_LJ



