""" Example of a Simulation using rumdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensembl with Lees-Edwards boundary conditions

"""
import numpy as np
import rumdpy as rp
import matplotlib.pyplot as plt

# Setup configuration: FCC Lattice
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)

configuration.simbox = rp.Simbox_LeesEdwards(configuration.D, configuration.simbox.lengths)


compute_plan = rp.get_default_compute_plan(configuration)
print(compute_plan)
compute_plan['gridsync'] = True # False #

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator to melt the crystal
dt = 0.005
num_blocks = 50
steps_per_block = 2048
running_time = dt*num_blocks*steps_per_block
temperature_low = 0.700

Ttarget_function = rp.make_function_ramp(value0=2.000, x0=running_time*(1/8), 
                                         value1=temperature_low, x1=running_time*(7/8))
integrator_NVT = rp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)



# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim_NVT = rp.Simulation(configuration, pairpot, integrator_NVT,
                    num_blocks=num_blocks, steps_per_block=steps_per_block,
                    storage='cool.h5', compute_plan=compute_plan)


calc_rdf = rp.CalculatorRadialDistribution(configuration, num_bins=1000)

for block in sim_NVT.blocks():
    print(block)
    print(sim_NVT.status(per_particle=True))
    calc_rdf.update()

rdf = calc_rdf.read()

print("Now run SLLOD simulation on what should now be a glass or polycrystal")

sc_output = 32

sr = 0.005
dt = 0.005
integrator_SLLOD = rp.integrators.SLLOD(shear_rate=sr, dt=dt)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim_SLLOD = rp.Simulation(configuration, pairpot, integrator_SLLOD,
                    num_blocks=100, steps_per_block=2048, scalar_output=sc_output,
                    storage='memory', compute_stresses=True, compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim_SLLOD.blocks():
    print(sim_SLLOD.status(per_particle=True))
    configuration.simbox.copy_to_host()
    box_shift = configuration.simbox.box_shift
    lengths = configuration.simbox.lengths
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim_SLLOD.summary())


u = sim_SLLOD.output['scalars'][:,:,0].flatten()/configuration.N
k = sim_SLLOD.output['scalars'][:,:,4].flatten()/configuration.N
sxy = sim_SLLOD.output['scalars'][:,:,9].flatten()/2./configuration.get_volume()
times = np.arange(len(u)) * sc_output *  dt
strains = times * sr

sxy_mean = np.mean(sxy)
print(f'{sr:.2g} {sxy_mean:.6f}')

plt.figure(1)
#plt.plot(strains, k)
#plt.plot(time, u)
#plt.figure(2)
plt.plot(strains, sxy)
plt.show()

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
