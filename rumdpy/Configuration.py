import numpy as np
import numba
import math
from numba import cuda
from rumdpy.colarray import colarray
from rumdpy.Simbox import Simbox

class Configuration:
    """ The configuration class

    Store particle vectors (positions, velocities, forces) and scalars (energy, virial, ...).
    Also store particle type and mass, simulation box dimensions, and image coordinates.

    Examples
    --------

    >>> import rumdpy as rp
    >>> conf = rp.Configuration(10, 3, np.array([10, 10, 10]))
    >>> print(conf.vector_columns)  # Print names of vector columns
    ['r', 'v', 'f', 'r_ref']
    >>> print(conf.sid) # Print names of scalar columns
    {'u': 0, 'w': 1, 'lap': 2, 'm': 3, 'k': 4, 'fsq': 5}
    >>> print(type(conf['r']))  # conf['r'] is a numpy array
    <class 'numpy.ndarray'>
    >>> print(conf['r'][0])  # Print position of particle 0
    [0. 0. 0.]
    >>> conf['r'] = np.ones((10, 3))  # Set positions to (1, 1, 1)
    >>> print(conf['r'][0])
    [1. 1. 1.]
    >>> print(conf['u'][0])  # Print potential energy of particle 0
    0.0
    """
    # vid = {'r':0, 'v':1, 'f':2, 'r_ref':3} # Superseeded by self.vector_columns
    sid = {'u': 0, 'w': 1, 'lap': 2, 'm': 3, 'k': 4, 'fsq': 5}
    num_cscalars = 3  # Number of scalars to be updated by force calculator. Avoid this!

    def __init__(self, N: int, D: int, simbox_lengths: np.ndarray, compute_stresses=True, ftype=np.float32, itype=np.int32) -> None:
        self.N = N
        self.D = D
        self.compute_stresses = compute_stresses
        self.vector_columns = ['r', 'v', 'f', 'r_ref']  # Should be user modifiable. Move r_ref to nblist
        if self.compute_stresses:
            self.vector_columns += ['sx', 'sy', 'sz'] # D=3 ASSUMED HERE!!!!
        # self.vectors = np.zeros((len(self.vid), N, D), dtype=ftype)
        self.vectors = colarray(self.vector_columns, size=(N, D), dtype=ftype)
        self.scalars = np.zeros((N, len(self.sid)), dtype=ftype)
        self.r_im = np.zeros((N, D), dtype=itype) # Move to vectors
        self.ptype = np.zeros(N, dtype=itype)     # Move to scalars
        self.ptype_function = self.make_ptype_function()
        self.simbox = Simbox(D, simbox_lengths)
        return

    def __setitem__(self, key, data):
        if key in self.vectors.column_names:
            self.set_vector(key, data)
        else:
            self.set_scalar(key, data)  # Improve error handling if key in neither
        return

    def __getitem__(self, key):
        if key in self.vectors.column_names:
            # return self.vectors[self.vid[key]]
            return self.vectors[key]
        return self.scalars[:,self.sid[key]]  # Improve error handling if key in neither

    def set_vector(self, key: str, data: np.ndarray) -> None:
        """ Set new vector data

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> conf.set_vector('r', np.ones((10, 3)))
        >>> print(conf['r'][0])
        [1. 1. 1.]
        """
        N, D = data.shape
        if key not in self.vector_columns:
            raise ValueError(f'Unknown vector column {key}. Try one of {self.vector_columns}')
        if N != self.N:
            raise ValueError(f'Inconsistent number of particles, {N} <> {self.N}')
        if D != self.D:
            raise ValueError(f'Inconsistent number of dimensions, {D} <> {self.D}')
        self.vectors[key] = data
        return

    def get_vector(self, key: str) -> np.ndarray: # Do we actually want a view instead of a copy (i.e. more like numpy)?
        """ Returns a copy of the vector lengths

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> print(conf.get_vector('r')[0])
        [0. 0. 0.]
        """
        if key not in self.vector_columns:
            raise ValueError(f'Unknown vector column {key}')
        idx = self.vector_columns.index(key)
        return self.vectors[self.vector_columns[idx]].copy()

    def set_scalar(self, key: str, data) -> None:
        """ Set new scalar data

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> print(conf['m'][0])
        0.0
        >>> conf.set_scalar('u', np.ones(10))
        >>> print(conf['m'][0])
        1.0
        """
        N, = data.shape
        if key not in self.sid:
            raise ValueError(f'Unknown scalar column {key}. Try one of {self.sid}')
        if N != self.N:
            raise ValueError(f'Inconsistent number of particles, {N} <> {self.N}')
        self.scalars[:, self.sid[key]] = data
        return

    def get_scalar(self, key: str): # Do we actually want a view instead of a copy (i.e. more like numpy)?
        """ Returns a copy of the scalar lengths

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> print(conf.get_scalar('m')[0])
        0.0
        """
        if key not in self.sid:
            raise ValueError(f'Unknown scalar column {key}. Try one of {self.sid}')
        idx = self.sid[key]
        return self.scalars[:, idx].copy()


    def copy_to_device(self):
        """ Copy all data to device memory

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> conf.copy_to_device()
        """
        # self.d_vectors = cuda.to_device(self.vectors)
        self.d_vectors = cuda.to_device(self.vectors.array)
        self.d_scalars = cuda.to_device(self.scalars)
        self.d_r_im = cuda.to_device(self.r_im)
        self.d_ptype = cuda.to_device(self.ptype)
        self.simbox.copy_to_device()
        return

    def copy_to_host(self):
        """ Copy all data to host memory

        Examples
        --------

        >>> import rumdpy as rp
        >>> conf = rp.Configuration(N=10, D=3, simbox_lengths=[10, 10, 10])
        >>> conf.copy_to_device()
        >>> # Do something with the configuration on the device
        >>> conf.copy_to_host()
        >>> print(conf['r'][0])
        [0. 0. 0.]
        """
        # self.vectors = self.d_vectors.copy_to_host()
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
        return self.simbox.volume(self.simbox.lengths)


    def randomize_velocities(self, T):
        if T > 0.0:
            self['v'] = generate_random_velocities(self.N, self.D, T=T, m=self['m'])
        else:
            self['v'] = np.zeros((self.N, self.D), np.float32)

    def set_kinetic_temperature(self, T, ndofs=None):
        if ndofs is None:
            ndofs = self.D * (self.N-1)

        T_ = np.sum( np.dot(self['m'], np.sum(self['v'] ** 2, axis=1)) ) / ndofs
        if T_ == 0:
            raise ValueError('Cannot rescale velocities when all equal to zero')
        self['v'] *= (T / T_) ** 0.5

# Helper functions

def generate_random_velocities(N, D, T, m=1, dtype=np.float32):
    v = np.zeros((N, D), dtype=dtype)
    for k in range(D):
        # to cover the case that m is a 1D array of length N, need to
        # generate one column at a time, passing the initial zeros as the
        # mean to avoid problems with inferring the correct shape
        v[:, k] = np.random.normal(v[:,k], (T/m)**0.5)
        CM_drift = np.mean(m*v[:, k]) / np.mean(m)
        v[:, k] -= CM_drift

    # rescale to get the kinetic temperature exactly right. The outer np.sum
    # is necessary when m is a scalar.
    T_ = np.sum( np.dot(m, np.sum(v ** 2, axis=1)) ) / (D * (N - 1))
    v *= (T / T_) ** 0.5
    return dtype(v)


@numba.njit
def generate_fcc_positions(nx, ny, nz, rho, dtype=np.float32):
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

    positions, simbox_data = generate_fcc_positions(nx, ny, nz, rho)
    N_, D = positions.shape
    if N==None:
        N = N_
    else:
        if N > N_:
            raise ValueError(f'N ({N}) needs to be equal to or smaller than number of particle in generated crystal ({N_})')
        scale_factor = (N/N_)**(1/3)
        positions *= scale_factor
        simbox_data *= scale_factor

    configuration = Configuration(N, D, simbox_data)
    configuration['r'] = positions[:N,:]
    configuration['m'] = np.ones(N, dtype=np.float32)  # Set masses
    configuration.ptype = np.zeros(N, dtype=np.int32)  # Set types

    return configuration


def configuration_to_lammps(conf, timestep=0) -> str:
    """ Convert a configuration to a string formatted as LAMMPS dump file
    >>> import rumdpy as rp
    >>> conf = rp.Configuration(4, 3, np.array([1, 1, 1]))
    >>> lammps_str = rp.configuration_to_lammps(conf, timestep=100)
    >>> # print(lammps_str, ofile=open('dump.lammps', 'w'))  # Uncomment to write to file
    """
    D = conf.D
    if D != 3:
        raise ValueError('Only 3D configurations are supported')
    masses = conf.get_scalar('m')
    positions = conf.get_vector('r')
    image_coordinates = conf.r_im
    forces = conf.get_vector('f')
    velocities = conf.get_vector('v')
    ptypes = conf.ptype
    simulation_box = conf.simbox.lengths

    # Header
    header = f'ITEM: TIMESTEP\n{timestep:d}\n'
    number_of_atoms = positions.shape[0]
    header += f'ITEM: NUMBER OF ATOMS\n{number_of_atoms:d}\n'
    header += f'ITEM: BOX BOUNDS pp pp pp\n'
    for k in range(3):
        header += f'{-simulation_box[k]/2:e} {simulation_box[k]/2:e}\n'
    # Atoms
    atom_data = 'ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz\n'
    for i in range(number_of_atoms):
        atom_data += f'{i + 1:d} {ptypes[i] + 1:d} {masses[i]:f} '
        for k in range(3):
            atom_data += f'{positions[i, k]:f} '
        for k in range(3):
            atom_data += f'{image_coordinates[i, k]:d} '
        for k in range(3):
            atom_data += f'{velocities[i, k]:f} '
        for k in range(3):
            atom_data += f'{forces[i, k]:f} '
        atom_data += '\n'
    # Combine header and atom lengths
    lammps_dump = header + atom_data
    return lammps_dump
