""" Example of a binary LJ simulation using rumdpy.

NVT simulation of the Kob-Andersen mixture, starting from a FCC crystal using a temperature ramp.

"""

import rumdpy as rp
import numpy as np

# Setup configuration: FCC crystal
configuration = rp.make_configuration_fcc(nx=6, ny=6, nz=6, rho=1.2, T=1.6)
configuration.ptype[::5] = 1     # Every fifth particle set to type 1 (4:1 mixture)
configuration['r'][27,2] += 0.01 # Pertube z-coordinate of particle 27

# Setup pair potential: Binary Kob-Andersen LJ mixture.
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
dt = 0.005 # timestep 
num_blocks = 32              # Do simulation in this many 'blocks'
steps_per_block = 2*32*1024  # ... each of this many steps
running_time = dt*num_blocks*steps_per_block
Ttarget_function = rp.make_function_ramp(value0=2.000, x0=running_time*(1/8), 
                                         value1=0.600, x1=running_time*(2/8))
integrator = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

sim = rp.Simulation(configuration, pairpot, integrator, 
                    num_blocks=num_blocks, steps_per_block=steps_per_block, 
                    storage='Data/KABLJ_Rho1.20_T0.600.h5') 

# Run simulation one block at a time
for block in sim.blocks():
    print(sim.status(per_particle=True))
print(sim.summary())

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 16 -o msd_KABLJ.pdf Data/KABLJ_Rho*T*.h5
