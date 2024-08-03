import numpy as np
import numba
import math
from numba import cuda
from rumdpy.colarray import colarray
from rumdpy.Simbox import Simbox

# IO
import h5py
import gzip


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

    >>> import rumdpy as rp
    >>> conf = rp.Configuration(D=3, N=1000)
    >>> print(conf.vector_columns)  # Print names of vector columns
    ['r', 'v', 'f', 'r_ref', 'sx', 'sy', 'sz']
    >>> print(conf.scalar_columns) # Print names of scalar columns
    ['u', 'w', 'lap', 'm', 'k', 'fsq']
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
    >>> conf = rp.Configuration(D=3)
    >>> conf['r'] = np.zeros((400, 3))
    >>> print(conf['r'].shape)
    (400, 3)

    """

    # vid = {'r':0, 'v':1, 'f':2, 'r_ref':3} # Superseeded by self.vector_columns
    sid = {'u': 0, 'w': 1, 'lap': 2, 'm': 3, 'k': 4, 'fsq': 5}
    num_cscalars = 3  # Number of scalars to be updated by force calculator. Avoid this!

    def __init__(self, D: int, N: int = None, compute_stresses=True, ftype=np.float32, itype=np.int32) -> None:
        self.D = D
        self.N = N
        self.compute_stresses = compute_stresses
        self.vector_columns = ['r', 'v', 'f', 'r_ref']  # Should be user modifiable. Move r_ref to nblist
        if self.compute_stresses:
            self.vector_columns += ['sx', 'sy', 'sz']  # D=3 ASSUMED HERE!!!!
        self.scalar_columns = list(self.sid.keys())
        self.simbox = None
        self.ptype_function = self.make_ptype_function()
        self.ftype = ftype
        self.itype = itype
        if self.N != None:
            self.__allocate_arrays()

    def __allocate_arrays(self):
        self.vectors = colarray(self.vector_columns, size=(self.N, self.D), dtype=self.ftype)
        self.scalars = np.zeros((self.N, len(self.sid)), dtype=self.ftype)
        self.r_im = np.zeros((self.N, self.D), dtype=self.itype)  # Move to vectors
        self.ptype = np.zeros(self.N, dtype=self.itype)  # Move to scalars
        return

    def __setitem__(self, key, data):
        if self.N is None:  # First time setting particle data, so allocate arrays
            if type(data) != np.ndarray:
                raise (TypeError)(
                    f'Number of particles, N, not determined yet, so assignment needs to be with a numpy array')
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

    def get_volume(self):
        """ Get volume of simulation box associated with configuration """
        return self.simbox.volume(self.simbox.lengths)

    def set_kinetic_temperature(self, T, ndofs=None):
        if ndofs is None:
            ndofs = self.D * (self.N - 1)

        T_ = np.sum(np.dot(self['m'], np.sum(self['v'] ** 2, axis=1))) / ndofs
        if T_ == 0:
            raise ValueError('Cannot rescale velocities when all equal to zero')
        self['v'] *= (T / T_) ** 0.5

    def randomize_velocities(self, T, seed=None, ndofs=None):
        """ Randomize velocities according to a given temperature. If T <= 0, set all velocities to zero. """
        if self.D is None:
            raise ValueError('Cannot randomize velocities. Start by assigning positions.')
        masses = self['m']
        if np.any(masses == 0):
            raise ValueError('Cannot randomize velocities when any mass is zero')
        if T > 0.0:
            self['v'] = generate_random_velocities(self.N, self.D, T=T, seed=seed, m=self['m'])
            # rescale to get the kinetic temperature exactly right
            self.set_kinetic_temperature(T=T, ndofs=ndofs)
        else:
            self['v'] = np.zeros((self.N, self.D), np.float32)

    def make_lattice(self, unit_cell: dict, cells: list, rho: float = None) -> None:
        """ Generate a lattice configuration

        The lattice is constructed by replicating the unit cell in all directions.
        Unit cell is a dictonary with `fractional_coordinates` for particles, and
        the `lattice_constants` as a list of lengths of the unit cell in all directions.

        Unit cells are avalible in rumdpy.unit_cells

        Example
        -------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(D=3)
        >>> conf.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
        >>> print(rp.unit_cells.FCC)  # Example of a unit cell dict
        {'fractional_coordinates': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], 'lattice_constants': [1.0, 1.0, 1.0]}

        """
        from rumdpy.tools import make_lattice
        positions, box_vector = make_lattice(unit_cell=unit_cell, cells=cells, rho=rho)
        self['r'] = positions
        self.simbox = Simbox(self.D, box_vector)
        return


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
    configuration.simbox = Simbox(D, simbox_data)
    configuration['m'] = np.ones(N, dtype=np.float32)  # Set masses
    configuration.ptype = np.zeros(N, dtype=np.int32)  # Set types

    return configuration


def configuration_to_hdf5(conf: Configuration, filename: str, meta_data=None):
    """ Write a configuration to a HDF5 file """
    if not filename.endswith('.h5'):
        filename += '.h5'
    with h5py.File(filename, "w") as f:
        f.attrs['simbox'] = conf.simbox.lengths
        if meta_data is not None:
            for item in meta_data:
                f.attrs[item] = meta_data[item]

        ds_r = f.create_dataset('r', shape=(conf.N, conf.D), dtype=np.float32)
        ds_v = f.create_dataset('v', shape=(conf.N, conf.D), dtype=np.float32)
        ds_p = f.create_dataset('ptype', shape=(conf.N), dtype=np.int32)
        ds_m = f.create_dataset('m', shape=(conf.N), dtype=np.float32)
        ds_r_im = f.create_dataset('r_im', shape=(conf.N, conf.D), dtype=np.int32)
        ds_r[:] = conf['r']
        ds_v[:] = conf['v']
        ds_p[:] = conf.ptype
        ds_m[:] = conf['m']
        ds_r_im[:] = conf.r_im


def configuration_from_hdf5(filename: str, reset_images=False) -> Configuration:
    """ Read a configuration from a HDF5 file """
    if not filename.endswith('.h5'):
        raise ValueError('Filename not inHDF5 format')
    with h5py.File(filename, "r") as f:
        lengths = f.attrs['simbox']
        r = f['r'][:]
        v = f['v'][:]
        ptype = f['ptype'][:]
        m = f['m'][:]
        r_im = f['r_im'][:]
    N, D = r.shape
    configuration = Configuration(D=D)
    configuration.simbox = Simbox(D, lengths)
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
    """ Write a configuration to a RUMD3 file """
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
            line_out = '%d %f %f %f %d %d %d %f %f %f\n' % (
                ptype[idx], r[idx, 0], r[idx, 1], r[idx, 2], r_im[idx, 0], r_im[idx, 1], r_im[idx, 2], v[idx, 0],
                v[idx, 1],
                v[idx, 2])
            f.write(line_out)


def configuration_from_rumd3(filename: str, reset_images=False) -> Configuration:
    """ Read a configuration from a RUMD3 file """
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

    configuration = Configuration(D=3)
    configuration.simbox = Simbox(3, lengths)
    configuration['r'] = r_array
    configuration['v'] = v_array
    configuration.r_im = im_array
    configuration.ptype = type_array
    configuration['m'] = m_array

    return configuration


def configuration_to_lammps(conf, timestep=0) -> str:
    """ Convert a configuration to a string formatted as LAMMPS dump file """
    D = conf.D
    if D != 3:
        raise ValueError('Only 3D configurations are supported')
    masses = conf['m']
    positions = conf['r']
    image_coordinates = conf.r_im
    forces = conf['f']
    velocities = conf['v']
    ptypes = conf.ptype
    simulation_box = conf.simbox.lengths

    # Header
    header = f'ITEM: TIMESTEP\n{timestep:d}\n'
    number_of_atoms = positions.shape[0]
    header += f'ITEM: NUMBER OF ATOMS\n{number_of_atoms:d}\n'
    header += f'ITEM: BOX BOUNDS pp pp pp\n'
    for k in range(3):
        header += f'{-simulation_box[k] / 2:e} {simulation_box[k] / 2:e}\n'
    # Atoms
    atom_data = 'ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz'
    for i in range(number_of_atoms):
        atom_data += f'\n{i + 1:d} {ptypes[i] + 1:d} {masses[i]:f} '
        for k in range(3):
            atom_data += f'{positions[i, k]:f} '
        for k in range(3):
            atom_data += f'{image_coordinates[i, k]:d} '
        for k in range(3):
            atom_data += f'{velocities[i, k]:f} '
        for k in range(3):
            atom_data += f'{forces[i, k]:f} '
        #atom_data += '\n'
    # Combine header and atom lengths
    lammps_dump = header + atom_data
    return lammps_dump
