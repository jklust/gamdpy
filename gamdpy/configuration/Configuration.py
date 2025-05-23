import numpy as np
import numba
import math
from numba import cuda
from .colarray import colarray
from ..simulation_boxes import Orthorhombic
from .topology import Topology, duplicate_topology, replicate_topologies
from ..simulation.get_default_compute_flags import get_default_compute_flags

# IO
import h5py
import gzip

# TODO: add possibility of "with ... as conf:" TypeError: 'Configuration' object does not support the context manager protocol

class Configuration:
    """ The configuration class

    Store particle vectors (positions, velocities, forces) and scalars (energy, virial, mass ...).
    Also store particle type, image coordinates, and the simulation box.

    Parameters
    ----------
    D : int
        Spatial dimension for the configuration.
    
    N : int [Optional]
        Number of particles. 
        If not set, this will be determined the first time particle data is written to the configuration. 
        
    Examples
    -------

    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3, N=1000)
    >>> print(conf.vector_columns)  # Print names of vector columns
    ['r', 'v', 'f']
    >>> print(conf.scalar_columns) # Print names of scalar columns
    ['U', 'W', 'K', 'm']
    >>> print(conf['r'].shape) # Vectors are stored as (N, D) numpy arrays
    (1000, 3)
    >>> print(conf['m'].shape) # Scalars are stored as (N,) numpy arrays
    (1000,)


    Data can be accessed via string keys (similar to dataframes in pandas):

    >>> conf['r'] = np.ones((1000, 3))
    >>> conf['v'] = 2   # Broadcast by numpy to correct shape
    >>> print(conf['r'] + 0.01*conf['v'])
    [[1.02 1.02 1.02]
     [1.02 1.02 1.02]
     [1.02 1.02 1.02]
     ...
     [1.02 1.02 1.02]
     [1.02 1.02 1.02]
     [1.02 1.02 1.02]]


    A configuration can be specified without setting the number particles, N.
    In that case N is determined the first time the particle data is written to the configuration:

    >>> import numpy as np
    >>> conf = gp.Configuration(D=3)
    >>> conf['r'] = np.zeros((400, 3))
    >>> print(conf['r'].shape)
    (400, 3)

    """

    scalar_parameters = ['m']
    scalar_computables_interactions = ['U', 'W', 'lapU']
    scalar_computables_integrator = ['K', 'Fsq']
    scalar_decriptions = {'m': 'Particle mass.',
                          'U': 'Potential energy.',
                          'W': 'Virial.',
                          'lapU': 'Laplace(U).',
                          'K': 'Kinetic energy.',
                          'Fsq': 'Squared length of force vector.', 
                          }


    def __init__(self, D: int, N: int = None, type_names=None, compute_flags=None, ftype=np.float32, itype=np.int32) -> None:
        self.D = D
        self.N = N

        self.type_names = type_names
        self.index_from_type_name = {}
        if type_names:
            for index, type_name in enumerate(type_names):
                self.index_from_type_name[type_name] = index

        self.compute_flags = get_default_compute_flags()
        if compute_flags != None:
            # only keys present in the default are processed
            for k in compute_flags:
                if k in self.compute_flags:
                    self.compute_flags[k] = compute_flags[k]
                else:
                    raise ValueError('Unknown key in compute_flags:%s' %k)

        self.vector_columns = ['r', 'v', 'f']  # Should be user modifiable
        if self.compute_flags['stresses']:
            if self.D > 4:
                raise ValueError("compute_flags['stresses'] should not be set for D>4")
            self.vector_columns += ['sx', 'sy', 'sz','sw'][:self.D]


        self.num_cscalars = 0
        self.sid = {}
        self.scalar_columns = []
        sid_index = 0

        for label in self.scalar_computables_interactions:
            if self.compute_flags[label]:
                self.sid[label] = sid_index
                self.scalar_columns.append(label)
                sid_index += 1
                self.num_cscalars += 1

        for label in self.scalar_computables_integrator:
            if self.compute_flags[label]:
                self.sid[label] = sid_index
                self.scalar_columns.append(label)
                sid_index += 1


        for label in self.scalar_parameters:
            self.sid[label] = sid_index
            self.scalar_columns.append(label)
            sid_index += 1

        self.simbox = None
        self.topology = Topology()
        self.ptype_function = self.make_ptype_function()
        self.ftype = ftype
        self.itype = itype
        if self.N != None:
            self.__allocate_arrays()

    def __allocate_arrays(self):
        self.vectors = colarray(self.vector_columns, size=(self.N, self.D), dtype=self.ftype)
        self.scalars = np.zeros((self.N, len(self.scalar_columns)), dtype=self.ftype)
        self.r_im = np.zeros((self.N, self.D), dtype=self.itype)  # Move to vectors
        self.ptype = np.zeros(self.N, dtype=self.itype)  # Move to scalars
        return

    def __repr__(self):
        return f'Configuration(D={self.D}, N={self.N}, compute_flags={self.compute_flags})'

    def __code__(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        code_str  = "# Define configuration class\n"
        code_str += f"from gamdpy import Configuration\n"
        code_str += f"configuration = Configuration(D={self.D}, N={self.N}, compute_flags={self.compute_flags})\n"
        # Following part needs to be done with a read function from the .h5
        for key in self.vector_columns:
            code_str += f"configuration['{key}'] = {self[key]}\n"
        for key in self.scalar_columns:
            code_str += f"configuration['{key}'] = {self[key]}\n"
        return code_str

    def __str__(self):
        if self.N == None:
            return f'{self.D} dimensional configuration. Particles not yet assigned.'
        str = f'{self.N} particles in {self.D} dimensions. Number density (atomic): {self.N/self.get_volume():.3f}'
        num_types = np.max(self.ptype)+1
        if num_types==1:
            str += '. Single component. '
        else:
            str += f'. {num_types} components with fractions '
            for ptype in range(num_types):
                str += f'{np.mean(self.ptype==ptype):.3f}, '
        str += '\nCurrent scalar data per particle:'
        for key in self.sid:
            str += f'\n{key+",":5} {np.mean(self.scalars[:,self.sid[key]]):.3f}'
            if key in self.scalar_decriptions:
                str += '\t' + self.scalar_decriptions[key]
        return str

    def __setitem__(self, key, data):
        if self.N is None:  # First time setting particle data, so allocate arrays
            if type(data) != np.ndarray:
                raise (TypeError)(
                    'Number of particles, N, not determined yet, so assignment needs to be with a numpy array')
            self.N = data.shape[0]
            self.__allocate_arrays()

        if key in self.vector_columns:
            self.__set_vector(key, data)
            return
        if key in self.scalar_columns:
            self.__set_scalar(key, data)
            return
        raise ValueError(f'Unknown key {key}. Vectors: {self.vector_columns}, Scalars: {self.scalar_columns}')

    def __set_vector(self, key: str, data: np.ndarray) -> None:
        """ Set new vector data """

        if type(data) == np.ndarray:  # Allow for possibility of using scalar float, which is then broadcast by numpy
            N, D = data.shape
            if N != self.N:
                raise ValueError(f'Inconsistent number of particles, {N} <> {self.N}')
            if D != self.D:
                raise ValueError(f'Inconsistent number of dimensions, {D} <> {self.D}')
        self.vectors[key] = data
        return

    def __set_scalar(self, key: str, data) -> None:
        """ Set new scalar data """

        if type(data) == np.ndarray:  # Allow for possibility of using scalar float, which is then broadcast by numpy
            N, = data.shape
            if N != self.N:
                raise ValueError(f'Inconsistent number of particles, {N} <> {self.N}')
        self.scalars[:, self.sid[key]] = data
        return

    def __getitem__(self, key):
        if key in self.vector_columns:
            return self.vectors[key]
        if key in self.scalar_columns:
            return self.scalars[:, self.sid[key]]
        raise ValueError(f'Unknown key {key}. Vectors: {self.vector_columns}, Scalars: {self.scalar_columns}')

    def copy_to_device(self):
        """ Copy all data to device memory """
        self.d_vectors = cuda.to_device(self.vectors.array)
        self.d_scalars = cuda.to_device(self.scalars)
        self.d_r_im = cuda.to_device(self.r_im)
        self.d_ptype = cuda.to_device(self.ptype)
        self.simbox.copy_to_device()
        return

    def copy_to_host(self):
        """ Copy all data to host memory """
        self.vectors.array = self.d_vectors.copy_to_host()
        self.scalars = self.d_scalars.copy_to_host()
        self.r_im = self.d_r_im.copy_to_host()
        self.ptype = self.d_ptype.copy_to_host()
        self.simbox.copy_to_host()
        return

    def make_ptype_function(self) -> callable:
        def ptype_function(pid, ptype_array):
            ptype = ptype_array[pid]  # Default: read from ptype_array
            return ptype

        return ptype_function

    def get_potential_energy(self) -> float:
        """ Get total potential energy of the configuration """
        return float(np.sum(self['U']))

    def get_volume(self):
        """ Get volume of simulation box associated with configuration """
        return self.simbox.get_volume()

    def set_kinetic_temperature(self, temperature, ndofs=None):
        if ndofs is None:
            ndofs = self.D * (self.N - 1)

        T_ = np.sum(np.dot(self['m'], np.sum(self['v'] ** 2, axis=1))) / ndofs
        if T_ == 0:
            raise ValueError('Cannot rescale velocities when all equal to zero')
        self['v'] *= (temperature / T_) ** 0.5

    def randomize_velocities(self, temperature, seed=None, ndofs=None):
        """ Randomize velocities according to a given temperature. If T <= 0, set all velocities to zero. """
        if self.D is None:
            raise ValueError('Cannot randomize velocities. Start by assigning positions.')
        masses = self['m']
        if np.any(masses == 0):
            raise ValueError('Cannot randomize velocities when any mass is zero')
        if temperature > 0.0:
            self['v'] = generate_random_velocities(self.N, self.D, T=temperature, seed=seed, m=self['m'])
            # rescale to get the kinetic temperature exactly right
            self.set_kinetic_temperature(temperature=temperature, ndofs=ndofs)
        else:
            self['v'] = np.zeros((self.N, self.D), np.float32)

    def make_lattice(self, unit_cell: dict, cells: list, rho: float = None) -> None:
        """ Generate a lattice configuration

        The lattice is constructed by replicating the unit cell in all directions.
        Unit cell is a dictonary with `fractional_coordinates` for particles, and
        the `lattice_constants` as a list of unit cell lengths in all directions.

        Unit cells are available in gamdpy.unit_cells

        Example
        -------

        >>> import gamdpy as gp
        >>> conf = gp.Configuration(D=3)
        >>> conf.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
        >>> print(gp.unit_cells.FCC)  # Example of a unit cell dict
        {'fractional_coordinates': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], 'lattice_constants': [1.0, 1.0, 1.0]}

        """
        from .make_lattice import make_lattice
        positions, box_vector = make_lattice(unit_cell=unit_cell, cells=cells, rho=rho)
        self['r'] = positions
        self.simbox = Orthorhombic(self.D, box_vector)
        return

    def make_positions(self, N, rho):
        """
        Generate particle positions in D dimensions.

        Positions are generated in a simple cubic configuration in D dimensions.
        Takes the number of particles N and the density rho as inputs

        Example
        -------

        >>> import gamdpy as gp
        >>> conf = gp.Configuration(D=3)
        >>> conf.make_positions(N=27, rho=0.2)
        """

        D = self.D
        part_per_line = np.ceil(pow(N, 1./D))

        box_length = pow(N/rho, 1./D)
        box_vector = np.array(D*[box_length])
        
        index = 0
        x = []      # empty list

        # This loop places particles in a simple cubic configuration
        # The first particle is in D*[0]
        while index < N:
            dcurrent = D - 1
            i_d = D*[float(0)]
            i_d[dcurrent] = (index / pow(part_per_line, dcurrent))
            rest = index % (pow(part_per_line, dcurrent))
            while dcurrent != 0:
                dcurrent = dcurrent - 1
                i_d[dcurrent] = (rest/pow(part_per_line,dcurrent))
                rest = index % (pow(part_per_line, dcurrent))
            x.append(i_d)
            index = index + 1
        pos = np.array(x)
        # Centering the array
        dcurrent = 1
        remove = 0
        while dcurrent < D:
            remove += D**(D-dcurrent)
            pos[:, dcurrent] -= remove/N
            dcurrent = dcurrent + 1
        pos -= np.array(D*[int(0.5*part_per_line)]) # center cube at 0
        # Scaling for density
        pos *= box_length/part_per_line
        # Saving to Configuration object
        self['r'] = pos
        self.simbox = Orthorhombic(self.D, box_vector)
        # Check all particles are in the box (-L/2, L/2)
        assert np.any(np.abs(pos))<0.5*box_length

        return
    
    def atomic_scale(self, density):
        actual_rho = self.N / self.get_volume()
        scale_factor = (actual_rho / density)**(1/3)
        self.vectors['r'] *= scale_factor
        self.simbox.lengths *= scale_factor


# Helper functions

def generate_random_velocities(N, D, T, seed, m=1, dtype=np.float32):
    """ Generate random velocities according to a given temperature. """
    v = np.zeros((N, D), dtype=dtype)
    # default value of seed is None and random.seed(None) has no effect
    np.random.seed(seed)
    for k in range(D):
        # to cover the case that m is a 1D array of length N, need to
        # generate one column at a time, passing the initial zeros as the
        # mean to avoid problems with inferring the correct shape
        v[:, k] = np.random.normal(v[:, k], (T / m) ** 0.5)
        CM_drift = np.mean(m * v[:, k]) / np.mean(m)
        v[:, k] -= CM_drift
    return dtype(v)

@numba.njit
def generate_fcc_positions(nx, ny, nz, rho, dtype=np.float32):
    # This function is not recommended to use, and should be considered deprecated
    # raise DeprecationWarning('Use Configuration.make_lattice() instead')

    D = 3
    conf = np.zeros((nx * ny * nz * 4, D), dtype=dtype)
    count = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                conf[count + 0, :] = [ix + 0.25, iy + 0.25, iz + 0.25]
                conf[count + 1, :] = [ix + 0.75, iy + 0.75, iz + 0.25]
                conf[count + 2, :] = [ix + 0.75, iy + 0.25, iz + 0.75]
                conf[count + 3, :] = [ix + 0.25, iy + 0.75, iz + 0.75]
                count += 4
    for k in range(D):
        conf[:, k] -= np.mean(conf[:, k])  # put sample in the middle of the box
    sim_box = np.array((nx, ny, nz), dtype=dtype)
    rho_initial = 4.0
    scale_factor = dtype((rho_initial / rho) ** (1 / D))

    return conf * scale_factor, sim_box * scale_factor


def make_configuration_fcc(nx, ny, nz, rho, N=None):
    """
    Generate Configuration for particle positions and simbox of a FCC lattice with a given density
    (nx x ny x nz unit cells), 
    and default types ('0') and masses ('1.')
    If N is given, only N particles will be in the configuration 
    (needs to be equal to or smaller than number of particle in generated crystal)
    """

    # This function is not recommended to use, and should be considered deprecated
    # raise DeprecationWarning('Use Configuration.make_lattice() instead')

    positions, simbox_data = generate_fcc_positions(nx, ny, nz, rho)
    N_, D = positions.shape
    if N == None:
        N = N_
    else:
        if N > N_:
            raise ValueError(
                f'N ({N}) needs to be equal to or smaller than number of particle in generated crystal ({N_})')
        scale_factor = (N / N_) ** (1 / 3)
        positions *= scale_factor
        simbox_data *= scale_factor

    configuration = Configuration(D=3)
    configuration['r'] = positions[:N, :]
    configuration.simbox = Orthorhombic(D, simbox_data)
    configuration['m'] = np.ones(N, dtype=np.float32)  # Set masses
    configuration.ptype = np.zeros(N, dtype=np.int32)  # Set types

    return configuration


def configuration_to_hdf5(configuration: Configuration, filename: str, meta_data=None) -> None:
    """ Write a configuration to a HDF5 file

    Parameters
    ----------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    filename : str
        filename of the output file .h5

    meta_data : str
        not used in the function so far (default None)

    Example
    -------

    >>> import os
    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3)
    >>> conf.make_positions(N=10, rho=1.0)
    >>> gp.configuration_to_hdf5(configuration=conf, filename="final.h5")
    >>> os.remove("final.h5")       # Removes file (for doctests)

    """

    if not filename.endswith('.h5'):
        filename += '.h5'
    with h5py.File(filename, "w") as f:
        f.attrs['simbox'] = configuration.simbox.lengths
        if meta_data is not None:
            for item in meta_data:
                f.attrs[item] = meta_data[item]

        ds_r = f.create_dataset('r', shape=(configuration.N, configuration.D), dtype=np.float32)
        ds_v = f.create_dataset('v', shape=(configuration.N, configuration.D), dtype=np.float32)
        ds_p = f.create_dataset('ptype', shape=(configuration.N), dtype=np.int32)
        ds_m = f.create_dataset('m', shape=(configuration.N), dtype=np.float32)
        ds_r_im = f.create_dataset('r_im', shape=(configuration.N, configuration.D), dtype=np.int32)
        ds_r[:] = configuration['r']
        ds_v[:] = configuration['v']
        ds_p[:] = configuration.ptype
        ds_m[:] = configuration['m']
        ds_r_im[:] = configuration.r_im


def configuration_from_hdf5(filename: str, reset_images=False, compute_flags=None) -> Configuration:
    """ Read a configuration from a HDF5 file

    Parameters
    ----------

    filename : str
        filename of the input file .h5

    reset_images : bool
        if True set the images to zero (deafult False)

    Returns
    -------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.configuration_from_hdf5("examples/Data/final.h5")
    >>> print(conf.D, conf.N, conf['r'][0])     # Print number of dimensions D, number of particles N and position of first particle
    3 10 [-0.7181449 -1.3644753 -1.5799187]

    """

    if not filename.endswith('.h5'):
        raise ValueError('Filename not in HDF5 format')
    with h5py.File(filename, "r") as f:
        lengths = f.attrs['simbox']
        r = f['r'][:]
        v = f['v'][:]
        ptype = f['ptype'][:]
        m = f['m'][:]
        r_im = f['r_im'][:]
    N, D = r.shape
    configuration = Configuration(D=D, compute_flags=compute_flags)
    configuration.simbox = Orthorhombic(D, lengths)
    configuration['r'] = r
    configuration['v'] = v
    configuration.ptype = ptype
    configuration['m'] = m
    if reset_images:
        configuration.r_im = np.zeros((N, D), dtype=np.int32)
    else:
        configuration.r_im = r_im
    return configuration


def configuration_to_rumd3(configuration: Configuration, filename: str) -> None:
    """ Write a configuration to a RUMD3 file 

    Parameters
    ----------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    filename : str
        filename of the output file .xyz.gz

    Example
    -------

    >>> import os
    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3)
    >>> conf.make_positions(N=10, rho=1.0)
    >>> gp.configuration_to_rumd3(configuration=conf, filename="restart.xyz.gz")
    >>> os.remove("restart.xyz.gz")       # Removes file (for doctests)

    """
    N = configuration.N
    if configuration.D != 3:
        raise ValueError("Only D==3 is compatibale with RUMD-3")

    r = configuration['r']
    v = configuration['v']
    ptype = configuration.ptype
    m = configuration['m']
    r_im = configuration.r_im

    num_types = max(ptype) + 1  # assumes consecutive types  starting from zero
    # find corresponding masses assuming unique mass for each type as required by RUMD-3
    masses = np.ones(num_types, dtype=np.float32)
    for type in range(num_types):
        type_first_idx = np.where(ptype == type)[0][0]
        masses[type] = m[type_first_idx]

    sim_box = configuration.simbox.lengths
    if not filename.endswith('.gz'):
        filename += '.gz'

    with gzip.open(filename, 'wt') as f:
        f.write('%d\n' % N)
        comment_line = 'ioformat=2 numTypes=%d' % (num_types)
        comment_line += ' sim_box=RectangularSimulationBox,%f,%f,%f' % (sim_box[0], sim_box[1], sim_box[2])
        comment_line += ' mass=%f' % (masses[0])
        for mass in masses[1:]:
            comment_line += ',%f' % mass
        comment_line += ' columns=type,x,y,z,imx,imy,imz,vx,vy,vz'
        comment_line += '\n'
        f.write(comment_line)
        for idx in range(N):
            line_out = '%d %.9f %.9f %.9f %d %d %d %f %f %f\n' % (
                ptype[idx], r[idx, 0], r[idx, 1], r[idx, 2], r_im[idx, 0], r_im[idx, 1], r_im[idx, 2], v[idx, 0],
                v[idx, 1],
                v[idx, 2])
            f.write(line_out)


def configuration_from_rumd3(filename: str, reset_images=False, compute_flags=None) -> Configuration:
    """ Read a configuration from a RUMD3 file 

    Parameters
    ----------

    filename : str
        filename of the output file .xyz.gz

    Returns
    -------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.configuration_from_rumd3("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles/restart0000.xyz.gz")
    >>> print(conf.D, conf.N, conf['r'][0])     # Print number of dimensions D, number of particles N and position of first particle
    3 4000 [ 7.197245   6.610052  -4.7467813]

    """
    with gzip.open(filename) as f:
        line1 = f.readline().decode()
        N = int(line1)

        line2 = f.readline().decode()
        meta_data_items = line2.split()
        meta_data = {}
        for item in meta_data_items:
            key, val = item.split("=")
            meta_data[key] = val

        num_types = int(meta_data['numTypes'])
        masses = [float(x) for x in meta_data['mass'].split(',')]
        assert len(masses) == num_types
        if meta_data['ioformat'] == '1':
            lengths = np.array([float(x) for x in meta_data['boxLengths'].split(',')], dtype=np.float32)
        else:
            assert meta_data['ioformat'] == '2'
            sim_box_data = meta_data['sim_box'].split(',')
            sim_box_type = sim_box_data[0]
            sim_box_params = [float(x) for x in sim_box_data[1:]]
            assert sim_box_type == 'RectangularSimulationBox'
            lengths = np.array(sim_box_params)
        # TO DO: handle LeesEdwards sim box
        assert meta_data['columns'].startswith('type,x,y,z,imx,imy,imz')
        has_velocities = (meta_data['columns'].startswith('type,x,y,z,imx,imy,imz,vx,vy,vz'))
        type_array = np.zeros(N, dtype=np.int32)
        r_array = np.zeros((N, 3), dtype=np.float32)
        im_array = np.zeros((N, 3), dtype=np.int32)
        v_array = np.zeros((N, 3), dtype=np.float32)
        m_array = np.ones(N, dtype=np.float32)

        for idx in range(N):
            p_data = f.readline().decode().split()
            ptype = int(p_data[0])
            type_array[idx] = ptype
            r_array[idx, :] = [float(x) for x in p_data[1:4]]
            if not reset_images:
                im_array[idx, :] = [int(x) for x in p_data[4:7]]
            if has_velocities:
                v_array[idx, :] = [float(x) for x in p_data[7:10]]
            m_array[idx] = masses[ptype]

    configuration = Configuration(D=3, compute_flags=compute_flags)
    configuration.simbox = Orthorhombic(3, lengths)
    configuration['r'] = r_array
    configuration['v'] = v_array
    configuration.r_im = im_array
    configuration.ptype = type_array
    configuration['m'] = m_array

    return configuration


def configuration_to_lammps(configuration, timestep=0) -> str:
    """ Convert a configuration to a string formatted as LAMMPS dump file 

    Parameters
    ----------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    timestep : float
        time at which the configuration is saved

    Returns
    -------

    str
        string formatted as LAMMPS dump file

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3)
    >>> conf.make_positions(N=10, rho=1.0)
    >>> lmp_dump = gp.configuration_to_lammps(configuration=conf)

    """
    D = configuration.D
    if D != 3 and D!=2:
        raise ValueError('Only 3D and 2D configurations are supported')
    masses = configuration['m']
    positions = configuration['r']
    image_coordinates = configuration.r_im
    forces = configuration['f']
    velocities = configuration['v']
    ptypes = configuration.ptype
    simulation_box = configuration.simbox.lengths

    # Header
    header = f'ITEM: TIMESTEP\n{timestep:d}\n'
    number_of_atoms = positions.shape[0]
    header += f'ITEM: NUMBER OF ATOMS\n{number_of_atoms:d}\n'
    header += f'ITEM: BOX BOUNDS pp pp pp\n'
    for k in range(D):
        header += f'{-simulation_box[k] / 2:e} {simulation_box[k] / 2:e}\n'
    if D==2:
        header += f'{-1 / 2:e} {1 / 2:e}\n'
    # Atoms
    atom_data = 'ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz'
    for i in range(number_of_atoms):
        atom_data += f'\n{i + 1:d} {ptypes[i] + 1:d} {masses[i]:f} '
        for k in range(D):
            atom_data += f'{positions[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{image_coordinates[i, k]:d} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{velocities[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{forces[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        #atom_data += '\n'
    # Combine header and atom lengths
    lammps_dump = header + atom_data
    return lammps_dump

def duplicate_molecule(topology, positions, particle_types, masses, cells, safety_distance, random_rotations=True):
    
    D=len(positions[0])
    num_molecules = 1
    for i in range(D):
        num_molecules *= cells[i]
    particles_per_per_molecule = len(positions)
    num_particles = num_molecules*particles_per_per_molecule
    configuration = Configuration(D=D, N=num_particles)
    configuration.topology = duplicate_topology(topology, num_molecules)

    positions_array = np.array(positions)

    positions_array -= np.min(positions_array, axis=0) # Shift positions: smallest coordinates = 0
    cell_length = np.max(positions_array) + safety_distance
    simbox_data = np.array(cells) * cell_length
    configuration.simbox = Orthorhombic(D, simbox_data)

    count = 0
    if D<3:
        cells = list(cells)
        cells.append(1)    
    for ix in range(cells[0]):
        for iy in range(cells[1]):
            for iz in range(cells[2]):
                arr = np.arange(D)
                if random_rotations:
                    np.random.shuffle(arr)
                configuration['r'][count:count+particles_per_per_molecule,0] = positions_array[:,arr[0]] + ix*cell_length
                configuration['r'][count:count+particles_per_per_molecule,1] = positions_array[:,arr[1]] + iy*cell_length
                if D>2:
                    configuration['r'][count:count+particles_per_per_molecule,2] = positions_array[:,arr[2]] + iz*cell_length
                configuration['m'][count:count+particles_per_per_molecule] = masses
                configuration.ptype[count:count+particles_per_per_molecule] = particle_types
                count += particles_per_per_molecule
    for i in range(configuration.D):
        configuration['r'][:,i] -= configuration.simbox.lengths[i]/2

    return configuration

def replicate_molecules(mol_topology_list, mol_positions_list, particle_type_list, masses_list, num_molecules_each_type_list, safety_distance, random_rotations=True, compute_flags=None):
    """ Construct a configuration containing different molecules, with the numbers of each type specified

        Parameters:
            mol_topology_list (list): A list of topologies, one for each molecule type to be replicated
            mol_positions_list (list): A list of position-lists, one position list for each molecule type, where a position list is a NxD array but as a list
            particle_type_list (list): A list of type_lists, where each type-list is a list of the atom types for that molecule type
            masses_list (list): A list of mass-lists, where each mass list is a list of atom masses for that molecule type
            num_molecules_each_type_list (list): A list of integers, specifying how many molecules of each type are to be included
            safety_distance (float): A length to be added in all directions to the size of the bounding box to be used  for each molecule when placing them initially on a lattice
            random_rotation (Bool): Whether the x,y,z coordinates in each molecule should be randomly permutated to give a simple randomization of orientations.
        Returns:
            configuration (Configuration): the resulting configuration with all molecules replicated and including the corresponding replicated topology
    """
    D = 3
    num_molecule_types = len(mol_topology_list)
    total_num_particles = 0
    total_num_molecules = 0
    mol_types = []
    positions_array_list = []
    cell_length_list = []
    size_molecule_type_list = []

    for idx in range(num_molecule_types):
        num_mol_this_type = num_molecules_each_type_list[idx]
        total_num_particles += len(mol_positions_list[idx]) * num_mol_this_type
        total_num_molecules += num_mol_this_type
        mol_types += [idx] * num_mol_this_type
        positions_array = np.array(mol_positions_list[idx])
        positions_array -= np.min(positions_array, axis=0)
        positions_array_list.append(positions_array)
        size_molecule_type_list.append( len(mol_positions_list[idx]) )
        cell_length = np.max(positions_array) + safety_distance
        cell_length_list.append(cell_length)

    # shuffle molecule types randomly. But commented out for now.
    np.random.shuffle(mol_types)


    configuration = Configuration(D=D, N=total_num_particles, compute_flags=compute_flags)
    configuration.topology = replicate_topologies(mol_topology_list, num_molecules_each_type_list, mol_types, size_molecule_type_list)

    max_cell_length = max(cell_length_list)
    # make a cubic box big enough to hold the total number of molecules
    num_cells_axis = math.ceil(total_num_molecules**(1/3))
    simbox_data = np.ones(D) * (num_cells_axis * max_cell_length)
    configuration.simbox = Orthorhombic(D, simbox_data)

    mol_count = 0
    particle_count = 0
    for ix in range(num_cells_axis):
        for iy in range(num_cells_axis):
            for iz in range(num_cells_axis):
                if mol_count < total_num_molecules:
                    # add a molecule
                    this_mol_type = mol_types[mol_count]
                    particles_this_molecule = size_molecule_type_list[this_mol_type]
                    arr = np.arange(D)
                    if random_rotations:
                        np.random.shuffle(arr)

                    configuration['r'][particle_count:particle_count+particles_this_molecule,0] = positions_array_list[this_mol_type][:,arr[0]] + ix*max_cell_length
                    configuration['r'][particle_count:particle_count+particles_this_molecule,1] = positions_array_list[this_mol_type][:,arr[1]] + iy*max_cell_length
                    configuration['r'][particle_count:particle_count+particles_this_molecule,2] = positions_array_list[this_mol_type][:,arr[2]] + iz*max_cell_length
                    configuration['m'][particle_count:particle_count+particles_this_molecule] = masses_list[this_mol_type]
                    configuration.ptype[particle_count:particle_count+particles_this_molecule] = particle_type_list[this_mol_type]
                    particle_count += particles_this_molecule

                    mol_count += 1

    assert mol_count == total_num_molecules
    assert particle_count == total_num_particles

    for i in range(configuration.D):
        configuration['r'][:,i] -= configuration.simbox.lengths[i]/2

    return configuration



def replicate_molecules2(molecule_dicts, num_molecules_each_type_list, safety_distance, random_rotations=True):
    """ Construct a configuration containing different molecules, with the numbers of each type specified

        Parameters:
            moleculde_dicts (list): A list of dictionaries, each of which contains keys "positions", "particle_types", "masses" and "topology", whose values are corresponding lists of data for that molecule
            num_molecules_each_type_list (list): A list of integers, specifying how many molecules of each type are to be included
            safety_distance (float): A length to be added in all directions to the size of the bounding box to be used  for each molecule when placing them initially on a lattice
            random_rotation (Bool): Whether the x,y,z coordinates in each molecule should be randomly permutated to give a simple randomization of orientations.
        Returns:
            configuration (Configuration): the resulting configuration with all molecules replicated and including the corresponding replicated topology
    """
    D = 3
    num_molecule_types = len(molecule_dicts)
    total_num_particles = 0
    total_num_molecules = 0
    mol_types = []
    positions_array_list = []
    cell_length_list = []
    size_molecule_type_list = []

    # unpack the list of molecule dictionaries and make lists of positions, particle_types, masses and topologies
    mol_topology_list = []
    mol_positions_list = []
    particle_type_list = []
    masses_list = []
    for idx in range(num_molecule_types):
        mol_positions_list.append(molecule_dicts[idx]["positions"])
        particle_type_list.append(molecule_dicts[idx]["particle_types"])
        masses_list.append(molecule_dicts[idx]["masses"])
        mol_topology_list.append(molecule_dicts[idx]["topology"])

    # tally the total numbers of particles and molecules, make shifted arrays of possitions for each molecule type
    for idx in range(num_molecule_types):
        num_mol_this_type = num_molecules_each_type_list[idx]
        total_num_particles += len(mol_positions_list[idx]) * num_mol_this_type
        total_num_molecules += num_mol_this_type
        mol_types += [idx] * num_mol_this_type
        positions_array = np.array(mol_positions_list[idx])
        positions_array -= np.min(positions_array, axis=0)
        positions_array_list.append(positions_array)
        size_molecule_type_list.append( len(mol_positions_list[idx]) )
        cell_length = np.max(positions_array) + safety_distance
        cell_length_list.append(cell_length)

    # shuffle molecule types randomly
    np.random.shuffle(mol_types)


    configuration = Configuration(D=D, N=total_num_particles)
    configuration.topology = replicate_topologies(mol_topology_list, num_molecules_each_type_list, mol_types, size_molecule_type_list)

    max_cell_length = max(cell_length_list)
    # make a cubic box big enough to hold the total number of molecules
    num_cells_axis = math.ceil(total_num_molecules**(1/3))
    simbox_data = np.ones(D) * (num_cells_axis * max_cell_length)
    configuration.simbox = Orthorhombic(D, simbox_data)

    mol_count = 0
    particle_count = 0
    for ix in range(num_cells_axis):
        for iy in range(num_cells_axis):
            for iz in range(num_cells_axis):
                if mol_count < total_num_molecules:
                    # add a molecule
                    this_mol_type = mol_types[mol_count]
                    particles_this_molecule = size_molecule_type_list[this_mol_type]
                    arr = np.arange(D)
                    if random_rotations:
                        np.random.shuffle(arr)

                    configuration['r'][particle_count:particle_count+particles_this_molecule,0] = positions_array_list[this_mol_type][:,arr[0]] + ix*max_cell_length
                    configuration['r'][particle_count:particle_count+particles_this_molecule,1] = positions_array_list[this_mol_type][:,arr[1]] + iy*max_cell_length
                    configuration['r'][particle_count:particle_count+particles_this_molecule,2] = positions_array_list[this_mol_type][:,arr[2]] + iz*max_cell_length
                    configuration['m'][particle_count:particle_count+particles_this_molecule] = masses_list[this_mol_type]
                    configuration.ptype[particle_count:particle_count+particles_this_molecule] = particle_type_list[this_mol_type]
                    particle_count += particles_this_molecule

                    mol_count += 1

    assert mol_count == total_num_molecules
    assert particle_count == total_num_particles

    for i in range(configuration.D):
        configuration['r'][:,i] -= configuration.simbox.lengths[i]/2

    return configuration
