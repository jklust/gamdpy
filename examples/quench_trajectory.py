""" Using gradient descent followed by conjugate gradient to 'quench' trajectory in a h5 file

The model is binary Kob & Andersen with shifted force cut-off, as found eg. in Data/KABLJ_Rho1.200_T0.400_toread.h5

Usage:
    python3 quench_trajectory.py filename
"""

import gamdpy as gp
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from scipy.optimize import minimize

Tconf_switch = 1e-4 # Do gradient descent until Tconf_switch is reached
include_cg = True   # ... and then do conjugate gradient if this flag is True
num_timeblocks = 2     # Number of timeblocks to quench, 0 to do all 
num_configurations = 0 # Numner of configurations in each timeblock, 0 to do all
gp.select_gpu()

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename
    else:
        filename = 'Data/KABLJ_Rho1.200_T0.400_toread.h5' # Used in testing
else:
    filename = 'Data/KABLJ_Rho1.200_T0.400_toread.h5' # Used in testing

# function to interface with minimize function from scipy
def calc_u(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate()
        return np.sum(configuration2['U'].astype('float64'))
def calc_du(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate()
        return -configuration2['f'].astype('float64').flatten()

# Load existing configuration, twice for convenience 
with h5py.File(filename, 'r') as f:
    configuration1 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
    configuration2 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
print(configuration1)
N = configuration1.N
D = configuration1.D

# Setup pair potential: Kob & Andersen Binary Lennard-Jones Mixture
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

evaluator = gp.Evaluator(configuration2, pair_pot)
evaluator.evaluate()
print(configuration2)

# Check that we are using the same model
assert(np.allclose(configuration1['U'], configuration2['U']))
assert(np.allclose(configuration1['W'], configuration2['W'], atol=0.00001)), f"({configuration1['W']} {configuration2['W']}"
assert(np.allclose(configuration1['f'], configuration2['f'], atol=0.0001)), f"({configuration1['f']} {configuration2['f']}"

# Setup integrator: NVT
integrator = gp.integrators.GradientDescent(dt=0.00001) # v = f*dt

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.ScalarSaver(16, compute_flags={'lapU':True}) ]

# Setup Simulation. 
sim = gp.Simulation(configuration1, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=2*1024,
                    storage='memory')

output = gp.tools.TrajectoryIO(filename).get_h5()
nblocks, nconfs, N, D = output['trajectory/positions'].shape

if num_timeblocks==0:
    num_timeblocks=nblocks

if num_configurations==0:
    num_configurations=nconfs

u_mins = np.zeros(( num_configurations, num_timeblocks))

quench_filename = filename[:-3]+'_quench.h5'
with h5py.File(quench_filename, "w") as f:
    configuration1.save(f, group_name="initial_configuration", mode="w")
    f.attrs['dt'] = output.attrs['dt']
    
    f.create_dataset('trajectory/positions',
        shape=(num_timeblocks, num_configurations, configuration1.N, configuration1.D),
        chunks=(1, 1, configuration1.N, configuration1.D),
        dtype=np.float32)
    f.create_dataset('trajectory/images',
        shape=(num_timeblocks, num_configurations, configuration1.N, configuration1.D),
        chunks=(1, 1, configuration1.N, configuration1.D),
        dtype=np.int32)

for saved_timeblock in range(num_timeblocks):

    for saved_conf in range(num_configurations):
        configuration1['r'] = gp.TrajectorySaver.extract_positions(output)[saved_timeblock, saved_conf]
        configuration2['r'] = gp.TrajectorySaver.extract_positions(output)[saved_timeblock, saved_conf]
        configuration1.r_im = gp.TrajectorySaver.extract_images(output)[saved_timeblock, saved_conf]
        configuration2.r_im = gp.TrajectorySaver.extract_images(output)[saved_timeblock, saved_conf]

        # Run simulation, i.e., gradient descent
        for timeblock in sim.run_timeblocks():
            Fsq_ = np.sum(configuration1['f'].astype(np.float64)**2)/configuration1.N
            lapU_ = np.sum(configuration1['lapU'].astype(np.float64))/configuration1.N
            Tconf_ = Fsq_ / lapU_
            if Tconf_ < Tconf_switch:
                break

        # Do conjugate gradient minimization as implemented in scipy    
        R0flat = configuration1['r'].astype('float64').flatten()
        res = minimize(calc_u, R0flat, method='CG', jac=calc_du, options={'gtol': 1e-9, 'disp': False, 'return_all':False})
        configuration2['r'] = res.x.reshape(N,D).astype('float32')
        
        u_min = calc_u(res.x)/N
        Fsq_min = np.sum(calc_du(res.x)**2)/N
        Tconf_min =  Fsq_min / np.sum(configuration2['lapU'].astype(np.float64)/N)

        print(f'{saved_timeblock=}, {saved_conf=}, {u_min=}, {Fsq_min=}, {Tconf_min=}') 

        u_mins[saved_conf, saved_timeblock] = u_min
        with h5py.File(quench_filename, "a") as f:
            f['trajectory/positions'][saved_timeblock,saved_conf,:,:] = configuration2['r']
            f['trajectory/images'][saved_timeblock,saved_conf,:,:] = configuration1.r_im  # GD was done on configuration1
        
plt.plot(u_mins, '.-')

plt.xlabel('Saved configuration')
plt.ylabel('Potential energy of inherent state')

if __name__ == "__main__":
    plt.show(block=False)

dynamics = gp.tools.calc_dynamics(output, 0, qvalues=[7.25, 5.5])
quench_output = gp.tools.TrajectoryIO(quench_filename).get_h5()
quench_dynamics = gp.tools.calc_dynamics(quench_output, 0, qvalues=[7.25, 5.5])
fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
axs[0].set_ylabel('MSD')
axs[1].set_ylabel('Intermediate scattering function')
axs[1].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
num_types = dynamics['msd'].shape[1]
for i in range(num_types):
    axs[0].loglog(dynamics['times'], dynamics['msd'][:,i], 'o-', label=f'True MSD, type:{i}')
    axs[0].loglog(quench_dynamics['times'], quench_dynamics['msd'][:,i], 'o--', label=f'Inherent MSD, Type:{i}')
    axs[1].semilogx(dynamics['times'], dynamics['Fs'][:,i], 'o-', label=f'True {i}, q={dynamics["qvalues"][i]}')    
    axs[1].semilogx(quench_dynamics['times'], quench_dynamics['Fs'][:,i], 'o--', label=f'Inherent {i}, q={dynamics["qvalues"][i]}')

axs[0].legend()
axs[1].legend()

if __name__ == "__main__":
    plt.show(block=True)
