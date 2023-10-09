import numpy as np
import rumdpy as rp
from numba import cuda
import pandas as pd

# Parameters determing how computations should be done. Should be collected in a 'compute_plan' or something...
UtilizeNIII = False
gridsync = True
pb = 8
tp = 16

# Generate a FCC lattice with a given density 
positions, simbox_data = rp.generate_fcc_positions(nx=4, ny=8, nz=8, rho=0.8442)
N, D = positions.shape

### Make configuration. Could be read from file, but this shows flexibility ###
c1 = rp.Configuration(N, D, simbox_data)
c1['r'] = positions
c1['v'] = rp.generate_random_velocities(N, D, T=1.44)
c1['m'] =  np.ones(N, dtype=np.float32)     # Set masses
c1.ptype = np.zeros(N, dtype=np.int32)      # Set types

# Make the pair potential
params = np.zeros((1,1), dtype="f,f,f")
params[0][0] = (4., -4., 2.5)
print('Pairpotential paramaters:\n', params)
LJ = rp.PairPotential(c1, rp.apply_shifted_force_cutoff(rp.LJ_12_6), UtilizeNIII=UtilizeNIII, params=params, max_num_nbs=1000)
num_cscalars = 3

# NOTE: following objects are specific to system size and other parameters for technical reasons

interactions = rp.make_interactions(c1, pb=pb, tp=tp,
                                    pairpotential_calculator=LJ.pairpotential_calculator,
                                    params_function=rp.params_function,
                                    num_cscalars=num_cscalars, 
                                    verbose=True, gridsync=gridsync, UtilizeNIII=False,)

integrator_step = rp.make_step_nve(c1, pb=pb, tp=tp, verbose=True, gridsync=gridsync)

integrate = rp.make_integrator(c1, integrator_step, interactions, pb=pb, tp=tp, verbose=True, gridsync=gridsync)

dt = np.float32(0.005)
skin = np.float32(0.5)

c1.copy_to_device()           
LJ.copy_to_device()
 
interaction_params = (LJ.d_params, skin, LJ.nblist.d_nblist,  LJ.nblist.d_nbflag)
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
if UtilizeNIII:
    df['u'] *= 2
    df['w'] *= 2
    df['lap'] *= 2
df['w'] *= 1/D/2
df['t'] = np.array(tt)  
    
rp.plot_scalars(df, N, D, figsize=(15,4))


