import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py
import os


def test_EAM_ZJW2004_Cu(main_dir=None):
    paramsCu = gp.EAM_ZJW_2004_params['Cu']

    # we follow LAMMPS and use sqrt(5) times the r_e parameter as the cutoff.
    cut = math.sqrt(5.) * paramsCu[0]
    paramsCu = np.append(paramsCu, cut)



    rho = 0.085 # number density in inverse cubic Angstrom
    TK = 2500 # Temperature in Kelvin


    Aa = 1.e-10 # Angstrom in m
    Na = 6.02214076e23
    gPerMole = 1.e-3/Na
    Da = 1.66053906892e-27
    eV = 1.602176634e-19
    kB = 1.380649e-23

    T = TK * kB/eV

    if main_dir is None:
        main_dir = os.path.join(os.getcwd(), "tests")

    reference_data_dir = os.path.join(main_dir, "reference_data")
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

    # Read reference data, saved from an equivalent LAMMPS simulations
    ref_data_path = os.path.join(reference_data_dir, "lammps_eam_Cu_ref.dat")

    ref_data = np.loadtxt(ref_data_path) # Note: ref_data has 513 rows, not 512

    diff_U = (U-ref_data[:-1,2])**2
    # sum the first 200 squared differences (representing 800 time steps)
    SSD = np.sum(diff_U[:200])
    
    assert SSD < 0.1
    # for later times the differences begin to grow as expected

if __name__ == '__main__':
    test_EAM_ZJW2004_Cu('.')


