""" Example of performing several simulation in one go using gamdpy.

An isomorph is traced out using the gamma method. The script demomstrates
the possibility of keeping the output of the simulation in memory (storage='memory').
This is usefull when a lot of short simulations are performed.

To plot the results do: 
python plot_isomorph_dynamics.py (generates: isomorph_dynamics.pdf)
python plot_isomorph_rdf.py (generates: isomorph_rdf.pdf)

For a simpler script performing multiple simulations, see isochore.py

"""

import pickle
import numpy as np
import gamdpy as gp

T = 2.00
rhos = [1.00, 1.05, 1.10, 1.15, 1.20, 1.00]
rhos = [1.00, 1.10, 1.20, 1.30, 1.40, 1.00]
rhos = [1.00, 1.50, 2.00, 2.50, 1.00]
data = []

# Setup fcc configuration
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=rhos[0])
configuration['m'] = 1.0

# Setup pair potential.
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
evaluator = gp.Evaluator(configuration=configuration, interactions=pair_pot)


# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.RestartSaver(),
                    gp.TrajectorySaver(),
                    gp.ScalarSaver(16),
                    gp.MomentumReset(100)]


evaluater = gp.Evaluator(configuration=configuration, interactions=pair_pot)

for index, rho in enumerate(rhos):

    # # Set temperature for simulation bases on Isomorph theory
    F1sq = np.sum(configuration['f']**2) 
    configuration.atomic_scale(density=rho)
    evaluator.evaluate()
    F2sq = np.sum(configuration['f']**2)

    if 0 < index < len(rhos)-1:
        T = round( (rhos[index-1]/rho)**(1/3) * (F2sq/F1sq)**(1/2) *  T, 3) # Force Method
        configuration.randomize_velocities(temperature=T)


    print(f'\nRho = {rho}, Temperature = {T}')

     # Setup integrator
    integrator = gp.integrators.NVT(temperature=T, tau=0.2/T**(1/2), dt=0.01/T**(1/2))

    # Setup Simulation
    sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                        num_timeblocks=16, steps_per_timeblock=512,
                        storage='memory') 
    
    # Setup on-the-fly calculation of Radial Distribution Function
    calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

    print('Equilibration:', end='\t')
    for block in sim.run_timeblocks():
        pass
    print(sim.status(per_particle=True))
    
    print('Production:', end='\t')
    for block in sim.run_timeblocks():
        calc_rdf.update()
    print(sim.status(per_particle=True))
    
    dynamics = gp.tools.calc_dynamics(sim.output, 0, qvalues=7.5*rho**(1/3))
    rdf = calc_rdf.read()
    data.append({'rho':rho, 'T':float(T), 'dynamics':dynamics, 'rdf':rdf})


with open('Data/isomorph.pkl', 'wb') as f:
    pickle.dump(data, f)

# To generat plots (isomorph_dynamics.pdf & isomorph_rdf.pdf) of data: 
# python plot_isomorph_dynamics.py; python plot_isomorph_rdf.py
