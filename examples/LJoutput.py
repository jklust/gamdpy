import numpy as np
import rumdpy as rp
from rumdpy.integrators import nve, nve_toxvaerd, nvt_nh, nvt_langevin, npt_langevin
import numba
from numba import cuda
import pandas as pd
import pickle
import json
import sys
import matplotlib.pyplot as plt

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

# Reproducing 'ATU' run in phd1 (LJ, N=864, rho=0.60, cooling from T=3.0 to 0.1 with dt=0.005, scalars: every 20'th step, trajectory: 194 log-blocks )    
# Timestep                                       200_000        2_000_000
# Rumd3 GeForce GTX TITAN, 5.5 TFlops, runtime: 56   sec        196   sec
# rumdpy RTX 3070 Laptop, 11.4 TFlops, runtime:  8.2 (4.7) sec   41.0 (37.1) sec (trajectories not yet included)


# Generate configuration with a FCC lattice
c1 = rp.make_configuration_fcc(nx=6,  ny=6,  nz=6,  rho=0.60,  T=3.44)  # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=0.8442,  T=2.44)  # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=8,  ny=8,  nz=8,  rho=1./0.9672,  T=1.44)  # N =  2*1024
#c1 = rp.make_configuration_fcc(nx=16,  ny=16,  nz=16,  rho=0.8442, T=1.44)  # N = 16*1024
c1.copy_to_device() 

pdict = {'N':c1.N, 'D':c1.D, 'simbox':c1.simbox.lengths, 'integrator':integrator, 'rdf':include_rdf}
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

inner_steps = 10_000
outer_steps = 200
steps_between_output = 20


T0 = rp.make_function_constant(value=3.0) # Not used for NVE*
P0 = rp.make_function_constant(value=4.0) # Not used for NV*
T1 = rp.make_function_ramp(value0=3.0, x0=0.0, value1=0.01, x1=dt*inner_steps*outer_steps)

def make_output_calculator(configuration, steps_between_output, compute_plan=compute_plan, verbose=False):
    D = configuration.D
    num_part = configuration.N
    pb = compute_plan['pb']
    tp = compute_plan['tp']
    gridsync = compute_plan['gridsync']
    UtilizeNIII = compute_plan['UtilizeNIII']
    num_blocks = (num_part-1)//pb + 1
    
    # Unpack indicies for vectors and scalars    
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indicies[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())
 
    
    def output_calculator(grid, vectors, scalars, r_im, sim_box,  output_array, step):
        """     
        """

        if step%steps_between_output==0:
            save_index = step//steps_between_output
        
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block * pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_part and my_t == 0:
                cuda.atomic.add(output_array, (save_index, 0), scalars[global_id][u_id])   # Potential energy
                cuda.atomic.add(output_array, (save_index, 1), scalars[global_id][w_id])   # Virial
                cuda.atomic.add(output_array, (save_index, 2), scalars[global_id][lap_id]) # Laplace
                cuda.atomic.add(output_array, (save_index, 3), scalars[global_id][fsq_id]) # F**2
                cuda.atomic.add(output_array, (save_index, 4), scalars[global_id][k_id])   # Kinetic energy
                if global_id ==0:
                    output_array[save_index,5] = vectors[r_id][global_id,0]   # x
                    output_array[save_index,6] = vectors[r_id][global_id,1]   # y
                    output_array[save_index,7] = vectors[r_id][global_id,2]   # z
            
        return
    return output_calculator

# Integrator for equilibration
integrate0, integrator_params0 = nvt.setup_output(c1, pairs['interactions'], None, T0, tau=0.2, dt=dt, compute_plan=compute_plan, verbose=False) 

output_calculator = make_output_calculator(c1, steps_between_output, compute_plan)

if integrator=='NVE':
    integrate, integrator_params = nve.setup_output(c1, pairs['interactions'], output_calculator, dt=dt, compute_plan=compute_plan, verbose=False)

if integrator=='NVE_Toxvaerd':
    integrate, integrator_params = nve_toxvaerd.setup_output(c1, pairs['interactions'], output_calculator, dt=dt, compute_plan=compute_plan, verbose=False)

if integrator=='NVT':
    integrate, integrator_params = nvt_nh.setup_output(c1, pairs['interactions'], output_calculator, T1, tau=0.2, dt=dt, compute_plan=compute_plan, verbose=False)
        
if integrator=='NVT_Langevin':
    integrate, integrator_params = nvt_langevin.setup_output(c1, pairs['interactions'], output_calculator, T1, alpha=0.1, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

if integrator=='NPT_Langevin':
    integrate, integrator_params = npt_langevin.setup_output(c1, pairs['interactions'],  output_calculator, T1, P0, alpha=0.1, alpha_baro=0.0001, 
        mass_baro=0.0001, volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2, dt=dt, seed=2023, compute_plan=compute_plan, verbose=False)

# Make rdf calculator
if include_rdf:
    num_bins = 500
    full_range=True
    gr_bins = np.zeros(num_bins, dtype=np.float64)
    d_gr_bins = cuda.to_device(gr_bins)
    host_array_zeros = np.zeros(d_gr_bins.shape, dtype=d_gr_bins.dtype)
    rdf_calculator = rp.make_rdf_calculator(c1, pair_potential = LJ, compute_plan=compute_plan, full_range=full_range, verbose=True)  

    
#  Run Simulation 
scalars_t = []
tt = []
vol_t = []

start = cuda.event()
end = cuda.event()
zero = np.float32(0.0)
rdf_count = 0

zero_output_array = np.zeros((inner_steps//steps_between_output, 10), dtype=np.float32)
d_output_array = cuda.to_device(zero_output_array)
integrate0(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params0, d_output_array, np.float32(0), inner_steps)

start.record() # Exclude first step from timing to get it more precise by excluding time for JIT compiling

for i in range(outer_steps):
    d_output_array = cuda.to_device(zero_output_array)

    #integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params, np.float32(i*inner_steps*dt), inner_steps)
    integrate(c1.d_vectors, c1.d_scalars, c1.d_ptype, c1.d_r_im, c1.simbox.d_data,  pairs['interaction_params'], integrator_params, d_output_array, np.float32(i*inner_steps*dt), inner_steps)
    
    scalars_t.append(d_output_array.copy_to_host())        
    
    c1.simbox.lengths = c1.simbox.d_data.copy_to_host()
    vol = (c1.simbox.lengths[0] * c1.simbox.lengths[1] * c1.simbox.lengths[2])
    vol_t.append(vol)

    if i>outer_steps//2 and include_rdf:
        rdf_count += 1
        rdf_calculator(c1.d_vectors, c1.simbox.d_data, c1.d_ptype, pairs['interaction_params'], d_gr_bins)
        temp_host_array = d_gr_bins.copy_to_host()       # offloading lengths from device and resetting decive array to zero.
        gr_bins += temp_host_array                       # ... (prevents overflow errors for longer runs)  
        d_gr_bins = cuda.to_device(host_array_zeros)

end.record()
end.synchronize()
timing_numba = cuda.event_elapsed_time(start, end)
nbflag = LJ.nblist.d_nbflag.copy_to_host()    
tps = outer_steps*inner_steps/timing_numba*1000
output_array = d_output_array.copy_to_host()
print(np.mean(output_array, axis=0))
print(output_array[0:20,:])

print('\tsteps :', outer_steps*inner_steps)
print('\tnbflag : ', nbflag)
print('\ttime :', timing_numba/1000, 's')
print('\tTPS : ', tps )
print('\tfinal box dims : ', c1.simbox.lengths[0], c1.simbox.lengths[1], c1.simbox.lengths[2])

scalars_t = np.concatenate(scalars_t)

# Save lengths
df = pd.DataFrame(scalars_t, columns=['u', 'w', 'lap', 'fsq', 'k', 'x0', 'y0', 'z0', 'd', 'd'])
df['t'] = np.array(np.arange(scalars_t.shape[0])*dt*steps_between_output)
df['vol'] = vol_t[0]
if integrator[0:3]!='NVE':
    df['Ttarget'] = numba.vectorize(T1)(np.array(df['t']))
if integrator=='NPT_Langevin':
    df['Ptarget'] = numba.vectorize(P0)(np.array(df['t'])) 
    #df['vol'] = np.array(vol_t)


print(df.mean())
df.to_csv('Data/LJ_scalars.csv')

if include_rdf:
    data = rp.normalize_and_save_gr(gr_bins, c1, pairs['interaction_params'], 
                                    full_range, rdf_count, filename='Data/LJ_rdf.dat')

# Plot results
#import analyze_LJ



