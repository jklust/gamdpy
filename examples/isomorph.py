""" Example of performing several simulation in one go using rumdpy.

An isomorph is traced out using the gamma method. The script demomstrates
the possibility of keeping the output of the simulation in memory (storage='memory').
This is usefull when a lot of short simulations are performed.

To plot the results do: 
python plot_isomorph_dynamics.pdf
python plot_isomorph_rdf.pdf

For a simpler script performing multiple simulations, see isochore.py

"""

import rumdpy as rp
import numpy as np
import pickle

# Setup pair potential.
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

T = 2.00
rhos = [1.00, 1.05, 1.10, 1.15, 1.20, 1.20]
data = []

for index, rho in enumerate(rhos):
    print(f'\nRho = {rho}, Temperature = {T}')

    # Setup fcc configuration
    configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=rho, T=2*T)
    
    # Setup integrator
    integrator = rp.integrators.NVT(temperature=T, tau=0.2, dt=0.0025)

    # Setup Simulation
    sim = rp.Simulation(configuration, pairpot, integrator, 
                        num_blocks=128, steps_per_block=512,
                        storage='memory') 
    
    # Setup on-the-fly calculation of Radial Distribution Function
    rdf_calculator = rp.RDF_calculator(configuration, num_bins=1000)

    print('Equilibration:', end='\t')
    for block in sim.blocks():
        pass
    print(sim.status(per_particle=True))
    
    print('Production:', end='\t')
    for block in sim.blocks():
        rdf_calculator.update()
    print(sim.status(per_particle=True))
    
    # Do data analysis
    U, W = rp.extract_scalars(sim.output, ['U', 'W'], first_block=1)
    dU = U - np.mean(U)
    dW = W - np.mean(W)
    gamma = np.dot(dW,dU)/np.dot(dU,dU)
    R = np.dot(dW,dU)/(np.dot(dW,dW)*np.dot(dU,dU))**0.5
    print(f'Gamma = {gamma:.3f},  R = {R:.3f}')

    dynamics = rp.tools.calc_dynamics(sim.output, 0, qvalues=7.5*rho**(1/3))
    rdf = rdf_calculator.read()
    data.append({'rho':rho, 'T':float(T), 'dynamics':dynamics, 'rdf':rdf})

    # Set temperature for next simulation
    if index+1<len(rhos)-1: 
        T = round((rhos[index+1]/rho)**gamma*T, 3) # Isomorph theory
    else:
        T = 2.00 # Last simulation Isothermal to first simulation

with open('Data/isomorph.pkl','wb') as f: pickle.dump(data, f)
# To generat plots (isomorph_dynamics.pdf & isomorph_rdf.pdf) of data: 
# python plot_isomorph_dynamics.py; python plot_isomorph_rdf.py
