""" Simple example of performing several simulation in one go using rumdpy.

Simulation of heating a Lennard-Jones crystal on an isochore in the NVT ensemble.
For an even simpler script, see minimal.py

"""

import rumdpy as rp
import numpy as np
import matplotlib.pyplot as plt

# Setup pair potential.
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

num_blocks = 16
steps_per_block = 1024*2

plt.figure(figsize=(12,6))
plt.xlabel('Reduced Time')
plt.ylabel('Reduced MSD')

old_rho = 1.00
old_T = 2.00
gamma = 1.0

for rho in [1.00, 1.05, 1.10, 1.15, 1.20]:

    T = (rho/old_rho)**gamma*old_T
    print(f'\n\n rho = {rho}, Temperature = {T}')

    # Setup fcc configuration
    configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=rho, T=2*T)
    
    # Setup integrator
    integrator = rp.integrators.NVT(temperature=T, tau=0.2, dt=0.005)

    # Setup Simulation
    sim = rp.Simulation(configuration, pairpot, integrator, 
                        num_blocks=num_blocks, steps_per_block=steps_per_block,
                        storage='memory') 

    print('Equilibration:')
    for block in sim.blocks():
        pass
    print(sim.status(per_particle=True))
    
    print('Production:')
    for block in sim.blocks():
        pass
    print(sim.status(per_particle=True))

    u = sim.output['scalars'][:,:,0].flatten()/configuration.N
    w = sim.output['scalars'][:,:,1].flatten()/configuration.N
    print(np.mean(u), np.mean(w))
    
    du = u - np.mean(u)
    dw = w - np.mean(w)

    gamma = np.dot(dw,du)/np.dot(du,du)
    R = np.dot(dw,du)/(np.dot(dw,dw)*np.dot(du,du))**0.5
    print(gamma, R)

    dynamics = rp.tools.calc_dynamics(sim.output,0)
    plt.loglog(dynamics['times']*rho**(1/3)*float(T)**0.5, dynamics['msd']*rho**(2/3), 
               'o--', label=f'rho={rho:.3f}, T={T:.3f}')
    
    old_rho = rho
    old_T = T

plt.legend()
plt.savefig('isomorph.pdf')
plt.show()



# To get a plot of the MSD do something like this:
# python3 -m rumdpy.tools.calc_dynamics -o Data/msd_r0.973.pdf Data/LJ_r0.973_T*.h5