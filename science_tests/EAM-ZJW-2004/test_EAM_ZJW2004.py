import os
import sys
import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py

import matplotlib.pyplot as plt

Aa = 1.e-10 # Angstrom in m
Na = 6.02214076e23
gPerMole = 1.e-3/Na
Da = 1.66053906892e-27
eV = 1.602176634e-19
kB = 1.380649e-23

rho = 0.085 # number density in inverse cubic Angstrom
TK = 2500 # Temperature in Kelvin
# convert temp to eV/kB units
T = TK * kB/eV

paramsCu = gp.EAM_ZJW_2004_params['Cu']

# we follow LAMMPS and use sqrt(5) times the r_e parameter as the cutoff.
cut = math.sqrt(5.) * paramsCu[0]
paramsCu = np.append(paramsCu, cut)


def test_EAM_ZJW2004_Cu(main_dir=None):


    if main_dir is None:
        main_dir = os.path.join(os.getcwd(), "science_tests", "EAM-ZJW-2004")

    reference_data_dir = main_dir # os.path.join(main_dir, "reference_data")
    conf_file_path = os.path.join(reference_data_dir, f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5")


    with h5py.File(conf_file_path) as f:
        conf = gp.Configuration.from_h5(f,"configuration")


    eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000)
    integrator = gp.NVE(dt=0.1)

    scalar_interval = 4

    runtime_actions = [gp.MomentumReset(100),
                       gp.ScalarSaver(scalar_interval)]

    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=2, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage='memory')

    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")


    U, W = gp.ScalarSaver.extract(sim.output, ['U', 'W'], per_particle=False, first_block=0)
    my_times = gp.ScalarSaver.get_times(sim.output, first_block=0)



    # Read reference data, saved from an equivalent LAMMPS simulations
    ref_data_path = os.path.join(reference_data_dir, "Data_lammps", "lammps_eam_Cu_ref.dat")


    ref_data = np.loadtxt(ref_data_path)
    U_ref = ref_data[:-1,2]  # Note: ref_data has 513 rows, not 512, hence the -1 in the row range.


    diff_U_sq = (U-U_ref)**2
    # sum the first 200 squared differences (representing 800 time steps)
    SSD = np.sum(diff_U_sq[:200])
    
    assert SSD < 0.1
    # for later times the differences begin to grow as expected
    # Make figure and save as pdf+png
    plt.figure(1)
    plt.plot(my_times, U, label="gamdpy")
    plt.plot(my_times, U_ref, label="LAMMPS")
    plt.xlabel("Time")
    plt.ylabel("Total potential energy (eV)")
    plt.savefig("eam_Cu_gamdy_vs_lammps.pdf")
    plt.savefig("eam_Cu_gamdy_vs_lammps.png")
    plt.legend()

# Move set up functionality here as a function, not a test function.

def setup():
    print("Setup procedure for EAM Cu test")
    print("Temperature in eV", T)


    conf = gp.Configuration(D=3)
    conf.make_lattice(gp.unit_cells.FCC, cells=[8,8,8], rho=rho)
    conf['m'] = 63.546
    conf.randomize_velocities(temperature=T)


    eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000)

    integrator = gp.NVT(temperature=T, tau=5.0, dt=0.1)

    runtime_actions = [gp.TrajectorySaver(),
                       gp.MomentumReset(100),
                       gp.ScalarSaver(16),
                       gp.RestartSaver()]

    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=15, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage='test_eam.h5')




    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")

    gamdpy_conf_filename = f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5"
    with h5py.File(gamdpy_conf_filename, "w") as f:
        conf.save(f, group_name='configuration', mode='w')

    print(sim.summary())


    dump_filename = f'Cu-liquid-rho{rho:.4f}-T{T:.4f}.lammps'
    if os.path.exists(dump_filename):
        os.remove(dump_filename)


    # when writing to LAMMPS dump file want to:
    # (1) increment the velocities by a half time step
    # (2) convert to LAMMPS "metal" units where the time unit is picosecond
    time_unit_in_picoseconds = Aa * math.sqrt(Da/eV)*1e12
    print('time unit', time_unit_in_picoseconds)
    unit_ratios = {'positions-box' : 1., 'velocities': 1./time_unit_in_picoseconds, 'forces':1.}

    dt = integrator.get_params(sim.configuration, sim.interactions_params)[0]
    #dt = 0.01 # for testing, to be consistent with the LAMMPS script
    print('Increment velocities by a half time-step for saving to LAMMPS dump file')
    print(f'dt={dt}')
    sim.configuration['v'] += (sim.configuration['f']/sim.configuration['m'][:, np.newaxis]) * (dt/2.)

    lmp_dump = gp.configuration_to_lammps(sim.configuration, timestep=0, unit_rescale=unit_ratios)
    print(lmp_dump, file=open(dump_filename, 'a'))

    print("Wrote gamdpy configuration file ", gamdpy_conf_filename)
    print("Wrote gamdpy configuration (dump) file ", dump_filename)
    print("If required move these files to Data_lammps to replace the previous versions")
    print("Note that the actual configurations will be different each time setup is called")


if __name__ == '__main__':
    if "setup" in sys.argv:
        print("Running Setup")
        setup()
    else:
        test_EAM_ZJW2004_Cu('.')
        plt.show()

