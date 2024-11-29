# This helper function is loading the output of a simulation as a dictionary following the formatting of sim.output
import sys
import numpy as np
import os

# This function is a wrapper for several possible inputs
class load_output():
    """ 
    This class handles loading and saving of simulation data.
    When the class is instanciated with an output from a previous simulation, the data are saved in self.h5 in the same format of the sim.output object.
    When the class is instanciated without input, self.h5 is None and can be assigned afterwards.

    Parameters
    ----------

    name : str
        Name of the file or folder to read output from. Can be a rumdpy .h output or a rumd3 TrajectoryFiles folder.

    Examples
    --------

    >>> import rumdpy as rp
    >>> import h5py
    >>> output = rp.tools.load_output("examples/Data/LJ_r0.973_T0.70.h5")                                   # Read from .h5
    Found .h5 file, loading to rumdpy as output dictionary
    >>> output = output.get_h5()
    >>> nblocks, nconfs, _ , N, D = output['block'].shape
    >>> print(f"Output file examples/Data/LJ_r0.973_T0.70.h5 containts a simulation of {N} particles in {D} dimensions")
    Output file examples/Data/LJ_r0.973_T0.70.h5 containts a simulation of 2048 particles in 3 dimensions
    >>> print(f"The simulation output is divided into {nblocks} blocks, each of them with {nconfs} configurations")
    The simulation output is divided into 8 blocks, each of them with 13 configurations
    >>> output = rp.tools.load_output("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles")    # Read from rumd3
    Found rumd3 TrajectoryFiles, loading to rumpdy as output dictionary
    >>> output = output.get_h5()
    >>> nblocks, nconfs, _ , N, D = output['block'].shape
    >>> print(f"File examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles containts a simulation of {N} particles in {D} dimensions")
    File examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles containts a simulation of 4000 particles in 3 dimensions
    >>> print(f"The simulation output is divided into {nblocks} blocks, each of them with {nconfs} configurations")
    The simulation output is divided into 2 blocks, each of them with 24 configurations

    """
    
    import h5py

    def __init__(self, name=""):
        import importlib.util           # This python library can be used if module can be imported

        ## Following lines defines the initialization behavior depending on given input
        if name[-3:]==".h5":
            print("Found .h5 file, loading to rumdpy as output dictionary")
            self.h5 = self.load_h5(name)
        elif "TrajectoryFiles" in name:
            print("Found rumd3 TrajectoryFiles, loading to rumpdy as output dictionary")
            if not os.path.isdir(name):
                print(f"Folder {name} doesn't exists")
                exit()
            self.h5 = self.load_rumd3(name)
        elif name=="":
            print("Warning: class initialized without input data, set self.h5 manually")
            self.h5 = None
        else:
            print("Input not recognized, unsupported format")
            self.h5 = None

        ## Following lines deals with the compression of the output
        # The class is initialized with some default compression settings depending on what is available.
        # These parameters can be changed by the user by changing the class attributes

        # Checks if hdf5plugin lib is available
        if importlib.util.find_spec("hdf5plugin")==None: 
            self.compression_type = "gzip"
            self.compression_opts = 6           # seems the best option in terms of compression and timing
        else:
            import hdf5plugin
            # if hdf5plugin is available use BZip2 compression (slightly better and faster)
            self.compression_type = hdf5plugin.BZip2()

    def get_h5(self) -> h5py.File:
        """ Returns self.h5 """
        return self.h5

    def load_h5(self, name:str) -> h5py.File:
        """ Makes self.h5 a view of the .h5 files """
        import h5py
        return h5py.File(name, "r")

    # Load from TrajectoryFiles (std rumd3 output)
    # It assumes trajectories are spaced in log2
    def load_rumd3(self, name:str) -> h5py.File:
        """ Reads a rumd3 TrajectoryFiles folder and convert it into rumdpy .h5 output. This function returns a memory .h5"""
        import h5py
        import os, gzip, glob
        import pandas as pd

        # Rumd3 output is always in D=3
        dim = 3
        # Checks if energy output is present
        if "LastComplete_energies.txt" not in os.listdir(name):
            print("LastComplete_energies.txt not present")
            energy = False
        else:
            energy = True

        # Checks if trajectory output is present
        if "LastComplete_trajectory.txt" not in os.listdir(name):
            print("LastComplete_trajectory.txt not present")
            traj = False
        else:
            traj = True

        # Exit if no output is found in TrajectoryFiles folder
        if not energy and not traj:
            print("No LastComplete file found, exiting")
            exit()

        # Reads block information from LastComplete_*
        if energy: nblocks, blocksize = np.loadtxt(f"{name}/LastComplete_energies.txt", dtype=np.int32)
        else     : nblocks, blocksize = np.loadtxt(f"{name}/LastComplete_trajectory.txt", dtype=np.int32)

        # Defining output memory .h5 file
        fullpath = f"{name}"
        output   = h5py.File(f"{id(fullpath)}.h5", "w", driver='core', backing_store=False)
        assert isinstance(output, h5py.File), "Error creating memory h5 file in load_output.load_rumd3"

        # Read trajectories
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

        # Read energies 
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
            if 'dt' in output.attrs.keys() : 
                output.attrs['steps_between_output'] = float(save_interval)/float(output.attrs['dt'])
            output.attrs['time_between_output'] = save_interval
            output.attrs['scalars_names'] = list(col_names)
            output.create_dataset('scalars', data=np.vstack(all_energies))

        return output

    def save_h5(self, name:str):
        """ 
        This method saves self.h5 to disk.
        It can be used to save sim.output to file if class is initialized without arguments and then self.h5 = sim.output . 
        By default output is compressed. This can be avoided setting self.compression_type = "gzip" and self.compression_opts=0 .
        """

        import h5py
        import importlib.util
        if importlib.util.find_spec("hdf5plugin")!=None:
            import hdf5plugin
        fout = h5py.File(name, "w") 
        fout.attrs.update(self.h5.attrs)
        for key in self.h5.keys():
            print(f"Writing dataset {key} to {name}")
            if importlib.util.find_spec("hdf5plugin")!=None:
                if self.compression_type == "gzip":
                    fout.create_dataset(key, data=self.h5[key], chunks=True, compression=self.compression_type, compression_opts=self.compression_opts, shuffle=True)
                else:
                    fout.create_dataset(key, data=self.h5[key], chunks=True, compression=self.compression_type, shuffle=True)
            elif self.compression_type == "gzip":
                fout.create_dataset(key, data=self.h5[key], chunks=True, compression=self.compression_type, compression_opts=self.compression_opts, shuffle=True)
            else:
                fout.create_dataset(key, data=self.h5[key], chunks=True, compression=self.compression_type)
            fout[key].attrs.update(self.h5[key].attrs)

        print(f"All written using {self.compression_type}")
        fout.close()
        print("Output file closed")
        return

if __name__ == '__main__':
    load_output(sys.argv[1])
