""" Test the class load_output. """

def test_load_output():
    import rumdpy as rp
    import h5py

    # Test read from h5
    output = rp.tools.load_output("examples/Data/LJ_r0.973_T0.70.h5")
    output = output.get_h5()
    isinstance(output.file, h5py.File)
    nblocks, nconfs, _ , N, D = output['block'].shape
    assert (N, D) == (2048, 3), "Error reading N and D from LJ_r0.973_T0.70.h5"

    # Test read from rumd3 TrajectoryFiles
    output = rp.tools.load_output("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles")
    output = output.get_h5()
    nblocks, nconfs, _ , N, D = output['block'].shape
    assert (N, D) == (4000, 3), "Error reading N and D from examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles"

    # Test read from rumd3 TrajectoryFiles, trajectory only
    output = rp.tools.load_output("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_trajonly")
    output = output.get_h5()
    assert isinstance(output.file, h5py.File), "Error with read from rumd3 trajectory only"

    # Test read from rumd3 TrajectoryFiles, energies only
    output = rp.tools.load_output("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_eneronly")
    output = output.get_h5()
    assert isinstance(output.file, h5py.File), "Error with read from rumd3 energy only"

    # Test initialization without input
    output = rp.tools.load_output()    
    output = output.get_h5()
    assert output == None, "Error with no input initialization"
    
    # Test read from unsupported format
    output = rp.tools.load_output("file.abc")                                                           
    output = output.get_h5()
    assert output == None, "Error with not recognized input/unsupported format"

if __name__ == '__main__':
    test_load_output()
