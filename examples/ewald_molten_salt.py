"""
This example highlights how to perform an Ewald convergence run of a molten salt system:
Hansen & McDonald, PRA 11, 2111 (1975)
The energy as a function of number of number-cutoff in reciprocal space is plotted.
"""

import gamdpy as gp
import matplotlib.pyplot as plt
import numpy as np

# System set-up
num_part = 1000
rho = 0.368
temperature = 0.0177
qA = -1
qB = 1
qs = [qA, qB]

# Setup configuration:
def make_conf(seed):
    configuration = gp.Configuration(D=3, compute_flags={'U': True, 'K': True})
    configuration.make_positions(N=num_part, rho=rho)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=temperature, seed=seed)
    configuration.ptype[::2] = 1
    configuration.set_charges_from_types(qs)
    return configuration

# Setup IPL potential
n = 9.0
vdw = gp.make_IPL_n(n=n)
eps = np.full((2,2), 1.0/n)
rc = 3.0
cut = np.full((2,2), rc)

pair_func = gp.apply_shifted_potential_cutoff(vdw)
pair_pot = gp.PairPotential(pair_func, params=[eps, cut], max_num_nbs=1000)

# Electrostatics parameters
damping = 1.0
rc_real = 5.0
cut_real = np.full((2,2), rc_real)


# Setting up simulation
dt = 0.02
num_timeblocks = 8 # Increase this to e.g. 32 for longer simulations and better statistics 
steps_per_timeblock = 512
runtime_actions = [gp.TrajectorySaver(), gp.RestartSaver(), gp.ScalarSaver(32), gp.MomentumReset(100)]
compute_plan = gp.get_default_compute_plan(make_conf(1))

integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

ns = [1, 6, 8, 10, 12, 14] # see Electrostatics.ncut
#ns = range(1, 17, 2) # Use this for a fuller exploration

Us = []

dyns = []
rdfs = []

for i, n in enumerate(ns):
    print('##########################')
    print(f'Summing over the first {n} shells of replicas...')
    configuration = make_conf(i)
    electro_pot = gp.Electrostatics(damping=damping, cutoff=cut_real, ncut=n, max_num_nbs=1000)
    sim = gp.Simulation(configuration, [pair_pot, electro_pot], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=compute_plan, storage="memory")
    # Equilibration
    print('Equilibrating.............')
    for block in sim.run_timeblocks():
        print(f'{sim.status(per_particle=True)}')

    # Setup on-the-fly calculation of Radial Distribution Function
    calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=140)


    # Production
    print('Running............')
    for block in sim.run_timeblocks():
        print(f'{sim.status(per_particle=True)}')
        calc_rdf.update()
    print(sim.summary())

    Us.append(np.mean(gp.ScalarSaver.extract(sim.output, ['U'], per_particle=True)) - electro_pot.self_energy)
    dyns.append(gp.calc_dynamics(sim.output, 0))
    rdfs.append(calc_rdf.read())

Us = np.array(Us)

plt.figure()
plt.plot(ns, Us/temperature, '.-')
plt.xlabel('Number of reciprocal shells')
plt.ylabel(r'$U/Nk_{\mathrm{B}}T$')
plt.show(block=False)

plt.figure()
for n, dyn in zip(ns, dyns):
    plt.loglog(dyn['times'], dyn['msd'][:,0], '+-', label=f'{n=}')
plt.xlabel('Time')
plt.ylabel('Mean Square Displacement')
plt.legend()
plt.show(block=False)

plt.figure()
for n, rdf in zip(ns, rdfs):
    plt.plot(rdf['distances'], rdf['rdf'][:,0,0], '+-', label=f'{n=}, AA')
for n, rdf in zip(ns, rdfs):
    plt.plot(rdf['distances'], rdf['rdf'][:,0,1], '+-', label=f'{n=}, AB')
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function')
plt.legend()
plt.show()


