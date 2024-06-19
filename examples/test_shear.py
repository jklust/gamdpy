""" Example of a Simulation using rumdpy, using explicit blocks.

Simulation of a Lennard-Jones crystal in the NVT ensembl with Lees-Edwards boundary conditions

"""

import rumdpy as rp
import matplotlib.pyplot as plt

# Setup configuration: FCC Lattice
configuration = rp.make_configuration_fcc(nx=8, ny=8, nz=8, rho=0.973, T=0.8 * 2)

configuration.simbox = rp.Simbox_LeesEdwards(configuration.D, configuration.simbox.lengths)


compute_plan = rp.get_default_compute_plan(configuration)
print(compute_plan)
compute_plan['gridsync'] = True # False

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = rp.PairPotential2(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.SLLOD(shear_rate=0.005, dt=0.005)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = rp.Simulation(configuration, pairpot, integrator,
                    num_blocks=16, steps_per_block=1024*4,
                    storage='memory', compute_stresses=True, compute_plan=compute_plan)

# Run simulation one block at a time
for block in sim.blocks():
    print(sim.status(per_particle=True))
    configuration.simbox.copy_to_host()
    box_shift = configuration.simbox.box_shift
    lengths = configuration.simbox.lengths
    print(f'box-shift={box_shift:.4f}, strain = {box_shift/lengths[1]:.4f}')
print(sim.summary())


u = sim.output['scalars'][:,:,0].flatten()/configuration.N

plt.plot(u)
plt.show()

# To get a plot of the MSD do something like this:
# python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5
