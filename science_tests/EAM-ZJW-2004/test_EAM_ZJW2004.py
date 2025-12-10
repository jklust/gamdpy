""" Science test for EAM.

Explanatino of modes:
    
    1. mode=="ci": run both setup and the test function(s), but a shortened version of both.
    If any files are written they should be deleted. Do not check agreement with reference data

    2. mode == "nightly": Don't run setup, only run the test files, but with full length, and use assertions to test for
    agreement with reference data. Do not keep output files.

    3. mode == "use": If "setup" is passed on the command-line then run the setup function with full duration. Otherwise
    run the test(s) with full duration. If running from the command line display graphs ie call plt.show(). Keep output files,
    which have temporary names (ending in .tmp) to avoid overwriting the version-controlled ones.

"""

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


def test_EAM_ZJW2004_Cu(mode):
    print("EAM (ZJW-2004) pure Cu test")
    print(f"Temperature in eV {T:.4}")
    # mode is "ci" or "nightly" or "use"
    if mode == "ci" or mode == "nightly":
        main_dir = os.path.join(os.getcwd(), "science_tests", "EAM-ZJW-2004")
    else:
        main_dir = "."

    reference_data_dir = main_dir # os.path.join(main_dir, "reference_data")
    conf_file_path = os.path.join(reference_data_dir, f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5")


    with h5py.File(conf_file_path) as f:
        conf = gp.Configuration.from_h5(f,"configuration")


    eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000)
    integrator = gp.NVE(dt=0.1)

    scalar_interval = 4

    runtime_actions = [gp.MomentumReset(100),
                       gp.ScalarSaver(scalar_interval)]

    num_timeblocks = {False:2, True:1}[mode=="ci"] # full length for "nightly" and "use"
    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=num_timeblocks, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage='memory')

    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")


    U, W = gp.ScalarSaver.extract(sim.output, ['U', 'W'], per_particle=False, first_block=0)
    my_times = gp.ScalarSaver.get_times(sim.output, first_block=0)



    # Read reference data, saved from an equivalent LAMMPS simulations
    ref_data_path = os.path.join(reference_data_dir, "Data_lammps", "lammps_eam_Cu_ref.dat")


    ref_data = np.loadtxt(ref_data_path)
    U_ref = ref_data[:-1,2]  # Note: ref_data has 513 rows, not 512, hence the -1 in the row range.

    if mode in ['use', 'nightly']:
        # in these cases we actually make an assert-based test
        diff_U_sq = (U-U_ref)**2
        # sum the first 200 squared differences (representing 800 time steps)
        SSD = np.sum(diff_U_sq[:200])
    
        assert SSD < 0.1
    # for later times the differences begin to grow as expected
    # Make figure and save as pdf+png (though when running as a test ie mode == "ci" or mode == "nightly")
    if mode == "use":
        plt.figure(1)
        plt.plot(my_times, U, label="gamdpy")
        plt.plot(my_times, U_ref, label="LAMMPS")
        plt.xlabel("Time")
        plt.ylabel("Total potential energy (eV)")
        plt.savefig("eam_Cu_gamdy_vs_lammps.pdf.tmp", format='pdf')
        plt.savefig("eam_Cu_gamdy_vs_lammps.png.tmp", format='png')
        plt.legend()
        print("Wrote graphical presentation of results as pdf and png files with extra suffix .tmp. \
              Remove the suffix by renaming if you wish to replace the version-ctronolled output files")

def setup(mode):
    print("Setup procedure for EAM Cu test")
    print(f"Temperature in eV {T:.4}")


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

    num_timeblocks = {False:15, True:2}[mode=="ci"] # full length for "nightly" and "use"
    output_filename = 'test_eam.h5'
    sim = gp.Simulation(conf, [eam_pot], integrator, num_timeblocks=num_timeblocks, steps_per_timeblock=1024, runtime_actions=runtime_actions, storage=output_filename)




    for block in sim.run_timeblocks():
        print(f"{sim.status(per_particle=True)}")
    print(sim.summary())

    if mode == "ci" or mode == "nightly":
        main_dir = os.path.join(os.getcwd(), "science_tests", "EAM-ZJW-2004")
    else:
        main_dir = "."

    gamdpy_conf_filename = dump_filename = os.path.join(main_dir, f"Cu-liquid-rho{rho:.4f}-T{T:.4f}.h5")
    with h5py.File(gamdpy_conf_filename+'.tmp', "w") as f:
        conf.save(f, group_name='configuration', mode='w')

    dump_filename = os.path.join(main_dir, f'Cu-liquid-rho{rho:.4f}-T{T:.4f}.lammps')

    # when writing to LAMMPS dump file want to:
    # (1) increment the velocities by a half time step
    # (2) convert to LAMMPS "metal" units where the time unit is picosecond
    time_unit_in_picoseconds = Aa * math.sqrt(Da/eV)*1e12
    print(f'time uni {time_unit_in_picoseconds:.6}')
    unit_ratios = {'positions-box' : 1., 'velocities': 1./time_unit_in_picoseconds, 'forces':1.}

    dt = integrator.get_params(sim.configuration, sim.interactions_params)[0]
    #dt = 0.01 # for testing, to be consistent with the LAMMPS script
    print('Increment velocities by a half time-step for saving to LAMMPS dump file')
    print(f'dt={dt:.6f}')
    sim.configuration['v'] += (sim.configuration['f']/sim.configuration['m'][:, np.newaxis]) * (dt/2.)

    lmp_dump = gp.configuration_to_lammps(sim.configuration, timestep=0, unit_rescale=unit_ratios)
    print(lmp_dump, file=open(dump_filename + '.tmp', 'a'))


    if mode == "ci" or mode == "nightly":
        # Could just not write the files, but it's probably good to test that code as well
        print("Removing gamdpy output file %s" % output_filename)
        os.remove(output_filename)
        print("Removing configuration files%s and %s"  %(gamdpy_conf_filename+'.tmp', dump_filename+'.tmp'))
        os.remove(gamdpy_conf_filename+'.tmp')
        os.remove(dump_filename+'.tmp')
    else:
        print("Wrote gamdpy configuration file ", gamdpy_conf_filename+'.tmp')
        print("Wrote gamdpy configuration (dump) file ", dump_filename+'.tmp')
        print("If it's necessary to replace the previous versions, remove the extension .tmp from these files and move the lammps file to Data_lammps ")
        print("Note that the actual configurations will be different each time setup is called")

mode = "ci" # default
if "nightly" in sys.argv:
    mode = "nightly"
if "use" in sys.argv:
    mode = "use" # if both present then go with "use"

print("Mode=%s" % mode)
if mode == "ci" or "setup" in sys.argv:
    setup(mode)


test_EAM_ZJW2004_Cu(mode)



if __name__ == '__main__':
    if "setup" not in sys.argv:
        plt.show()

