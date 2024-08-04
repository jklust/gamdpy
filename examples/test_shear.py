""" Example of a Simulation using rumdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensembl with Lees-Edwards boundary conditions

"""
import numpy as np
import rumdpy as rp
import matplotlib.pyplot as plt

run_NVT = False # True # 

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

temperature_low = 0.700
gridsync = True


if run_NVT:
    # Setup configuration: FCC Lattice
    configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973)
    configuration.randomize_velocities(T=2.0)

    # Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
    sim_NVT = rp.Simulation(configuration, pairpot, integrator_NVT,
                            num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                            steps_between_momentum_reset=100,
                            storage='cool.h5', compute_plan=compute_plan)
    


    # Setup integrator to melt the crystal
    dt = 0.005
    num_blocks = 10 # 50
    steps_per_block = 2048
    running_time = dt*num_blocks*steps_per_block


    Ttarget_function = rp.make_function_ramp(value0=2.000, x0=running_time*(1/8), 
                                             value1=temperature_low, x1=running_time*(7/8))
    integrator_NVT = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

    # Set simulation up. Total number of timesteps: num_blocks * steps_per_block
    sim_NVT = rp.Simulation(configuration, pairpot, integrator_NVT,
                            num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                            storage='cool.h5', compute_plan=compute_plan)


    calc_rdf = rp.CalculatorRadialDistribution(configuration, num_bins=1000)

    for block in sim_NVT.timeblocks():
        print(block)
        print(sim_NVT.status(per_particle=True))
        calc_rdf.update()

        rdf = calc_rdf.read()

    # save both in hdf5 and rumd-3 formats
    rp.configuration_to_hdf5(configuration, 'LJ_cooled_0.70.h5')
    rp.configuration_to_rumd3(configuration, 'LJ_cooled_0.70.xyz.gz')

else:
    configuration0 = rp.configuration_from_hdf5('LJ_cooled_0.70.h5')
    configuration1 = rp.configuration_from_rumd3('LJ_cooled_0.70.xyz.gz')
    configuration = configuration1

compute_plan = rp.get_default_compute_plan(configuration)
compute_plan['gridsync'] = gridsync
print("Now run SLLOD simulation on what should now be a glass or polycrystal")

sc_output = 1

sr = 0.005
dt = 0.01

configuration.simbox = rp.Simbox_LeesEdwards(configuration.D, configuration.simbox.lengths)

integrator_SLLOD = rp.integrators.SLLOD(shear_rate=sr, dt=dt)

# set the kinetic temperature to the exact value associated with the desired
# temperature since SLLOD uses an isokinetic thermostat
configuration.set_kinetic_temperature(temperature_low, ndofs=configuration.N*3-4) # remove one DOF due to constraint on total KE

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim_SLLOD = rp.Simulation(configuration, pairpot, integrator_SLLOD,
                          num_timeblocks=10, steps_per_timeblock=512, scalar_output=sc_output,
                          steps_between_momentum_reset=100,
                          storage='memory', compute_stresses=True, compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim_SLLOD.timeblocks():
    print(sim_SLLOD.status(per_particle=True))
    configuration.simbox.copy_to_host()
    box_shift = configuration.simbox.box_shift
    lengths = configuration.simbox.lengths
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim_SLLOD.summary())


u = sim_SLLOD.output['scalars'][:,:,0].flatten()/configuration.N
k = sim_SLLOD.output['scalars'][:,:,4].flatten()/configuration.N
sxy = sim_SLLOD.output['scalars'][:,:,9].flatten()/configuration.get_volume()
times = np.arange(len(u)) * sc_output *  dt
stacked_output = np.column_stack((times, u, k, sxy))
np.savetxt('shear_run.txt', stacked_output, delimiter=' ', fmt='%f')


strains = times * sr

sxy_mean = np.mean(sxy)
print(f'{sr:.2g} {sxy_mean:.6f}')

#plt.figure(1)
#plt.plot(strains, k)
#plt.plot(time, u)
#plt.figure(2)
#plt.plot(strains, sxy)
#plt.show()

# STRAINRATE VS MEAN STRESS


# 0.001 0.014872
# 0.0025 0.043739
# 0.005 0.069240
# 0.0075 0.103628
# 0.01 0.139017
# 0.02 0.253017
# 0.03 0.348075


# quadratic fit gives the following
# 0.0027409 + 14.438 * x - 97.323 * x^2

# The small value of the stress at zero, 0.0027409 is consistent with zero
# which is promising and we can read the Newtonian viscosity off as 14.438
