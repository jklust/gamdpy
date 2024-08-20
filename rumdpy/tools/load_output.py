# This helper function is loading the output of a simulation as a dictionary following the formatting of sim.output
import sys
import numpy as np

# This function is a wrapper for several possible inputs
def load_output(name : str) -> dict:
    """
    This function takes a saved simulation and returns it as sim.output object.

    Example
    -------

    >>> import rumdpy as rp
	>>> import h5py
    >>> output = rp.tools.load_output("examples/Data/LJ_r0.973_T0.70.h5") 
    Found .h5 file, loading to rumdpy as output dictionary
	>>> isinstance(output.file, h5py.File)
	True
    >>> nblocks, nconfs, _ , N, D = output['block'].shape
    >>> assert (N, D) == (2048, 3), "Error reading N and D from LJ_r0.973_T0.70.h5" 
    """

    if name[-3:]==".h5":
        print("Found .h5 file, loading to rumdpy as output dictionary")
        return load_h5(name)
    elif name=="TrajectoryFiles":
        print("Found rumd3 TrajectoryFiles, loading to rumpdy as output dictionary")
        return load_rumd3(name)
    else:
        print("Input not recognized")
        return

# Load from h5 (std rumpdy output)
def load_h5(name:str) -> dict:
    import h5py
    return h5py.File(name, "r")

# Load from TrajectoryFiles (std rumd3 output)
# It assumes trajectories are spaced in log2
def load_rumd3(name:str) -> dict:
    import os, gzip, glob
    import pandas as pd
    import h5py
    # Check what's there
    energy, traj  = True, True

    # Rumd3 output is always in D=3
    dim = 3
    if "LastComplete_energies.txt" not in os.listdir(name):
        print("LastComplete_energies.txt not present")
        energy = False
    if "LastComplete_trajectory.txt" not in os.listdir(name):
        print("LastComplete_trajectory.txt not present")
        traj = False
    if not energy and not traj:
        print("No LastComplete file found, exiting")
        exit()
    if not energy: nblocks, blocksize = np.loadtxt(f"{name}/LastComplete_energies.txt", dtype=np.int32)
    else         : nblocks, blocksize = np.loadtxt(f"{name}/LastComplete_trajectory.txt", dtype=np.int32)

    # Defining output memory .h5 file
    fullpath = f"{os.getcwd()}/{name}"
    output   = h5py.File(f"{id(fullpath)}.h5", "w", driver='core', backing_store=False)
    assert isinstance(output, h5py.File)

    # Read and copy trajectories
    if traj:
        traj_files = sorted(glob.glob(f"{name}/trajectory*"))
		# Remove last block if incomplete
        if traj_files[-1]==f"{name}/trajectory{nblocks+1:04d}.xyz.gz": traj_files = traj_files[:-1]
        # Read metadata from first file in the list
        with gzip.open(f"{traj_files[0]}", "r") as f:
            npart = int(f.readline())
            cmt_line = f.readline().decode().split()
            meta_data = dict()
            for item in cmt_line:
                key, val = item.split("=")
                meta_data[key] = val
            num_types = int(meta_data['numTypes'])
            masses = [float(x) for x in  meta_data['mass'].split(',')]
            assert len(masses) == num_types
            if meta_data['ioformat'] == '1':
                lengths = np.array([float(x) for x in meta_data['boxLengths'].split(',')], dtype=np.float32)
            else:
                assert meta_data['ioformat'] == '2'
                sim_box_data = meta_data['sim_box'].split(',')
                sim_box_type = sim_box_data[0]
                sim_box_params = [float(x) for x in sim_box_data[1:]]
                assert sim_box_type == 'RectangularSimulationBox'
                lengths = np.array(sim_box_params, dtype=np.float32)
                integrator_data = meta_data['integrator'].split(',')
                timestep = integrator_data[1]
        # Loop over the files and read them assuming each line is type, x, y, z, imx, imy, imz
        ntrajinblock = int(1+np.log2(blocksize))
        toskip1 = np.array([  (npart+2)*x for x in range(1+ntrajinblock)])
        toskip2 = np.array([1+(npart+2)*x for x in range(1+ntrajinblock)])
        toskip  = sorted(list(np.concatenate((toskip1, toskip2))))
        #print(toskip)
        positions = list()
        images    = list()
        for trajectory in traj_files:
            tmp_data   = pd.read_csv(trajectory, skiprows = toskip, usecols=[0,1,2,3,4,5,6], names=["type", "x", "y", "z", "imx", "imy", "imz"], delimiter=" ")
            type_array = tmp_data['type'].to_numpy()
            pos_array  = np.c_[tmp_data['x'].to_numpy(), tmp_data['y'].to_numpy(), tmp_data['z'].to_numpy()]
            img_array  = np.c_[tmp_data['imx'].to_numpy(), tmp_data['imy'].to_numpy(), tmp_data['imz'].to_numpy()]
            positions.append(pos_array.reshape((1+ntrajinblock,npart,dim)))
            images.append(pos_array.reshape((1+ntrajinblock,npart,dim)))
        # Saving data in output h5py
        output.attrs['dt'] =  timestep 
        output.attrs['simbox_initial'] = lengths 
        output.attrs['vectors_names'] = ["r", "r_im"]
        output.create_dataset("block", shape=(len(traj_files), 1+ntrajinblock, 2, npart, dim))
        output['block'][:,:,0,:,:] = np.array(positions) 
        output.create_dataset("ptype", data=type_array[:npart], shape=(npart), dtype=np.int32)

    # Read and copy trajectories
    if energy:
        energy_files = sorted(glob.glob(f"{name}/energies*"))
		# Remove last block if incomplete
        if energy_files[-1]==f"{name}/energies{nblocks+1:04d}.dat.gz": energy_files = energy_files[:-1]
        # Read metadata from first file in the list
        with gzip.open(f"{energy_files[0]}", "r") as f:
            # ioformat=2 N=4096 Dt=147.266830 columns=ke,pe,p,T,Etot,W
            cmt_line = f.readline().decode().split()[1:]
            meta_data = dict()
            for item in cmt_line:
                key, val = item.split("=")
                meta_data[key] = val
            npart = meta_data['N']
            save_interval = meta_data['Dt']
            col_names = meta_data['columns'].split(",")
        all_energies = list()
        for energies in energy_files:
            tmp_data   = pd.read_csv(energies, skiprows=1, names=col_names, usecols = [i for i in range(len(col_names))], delimiter=" ")
            all_energies.append(tmp_data.to_numpy())
        # Saving data in output h5py
        if output.attrs!=None : output.attrs['dt'] = timestep 
        output.attrs['steps_between_output'] = float(save_interval)/float(output.attrs['dt'])
        output.attrs['scalars_names'] = list(col_names)
        output.create_dataset('scalars', data=np.vstack(all_energies))

    return output

if __name__ == '__main__':
    load_output(sys.argv[1])
