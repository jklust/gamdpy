import numpy as np
import rumdpy as rp
import numba
import math
from numba import cuda


#############################################################
#### Radial Distribution Function  
#############################################################

class CalculatorWidomInsertion():
    """ Calculator class for the radial distribution function, g(r)

    Parameters
    ----------

    configuration : rumdpy.Configuration
        The configuration object for which the radial distribution function is calculated.

    num_bins : int
        The number of bins in the radial distribution function.

    compute_plan : dict

    Example
    -------

    >>> import rumdpy as rp
    >>> sim = rp.get_default_sim()
    >>> calc_rdf = rp.CalculatorRadialDistribution(sim.configuration, num_bins=1000)
    >>> for _ in sim.timeblocks():
    ...     calc_rdf.update()      # Current configuration to rdf
    >>> rdf_data = calc_rdf.read() # Read the rdf data as a dictionary
    >>> r = rdf_data['distances']  # Pair distances
    >>> rdf = rdf_data['rdf']      # Radial distribution function
    """

    def __init__(self, configuration, num_bins, compute_plan=None) -> None:
        self.configuration = configuration
        self.num_bins = num_bins
        self.count = 0  # How many times have statistics been added to?

        self.compute_plan = compute_plan
        if self.compute_plan is None:
            self.compute_plan = rp.get_default_compute_plan(configuration=configuration)

            # Allocate space for statistics
        self.rdf_list = []
        self.gr_bins = np.zeros(self.num_bins, dtype=np.float64)
        self.d_gr_bins = cuda.to_device(self.gr_bins)
        self.host_array_zeros = np.zeros(self.d_gr_bins.shape, dtype=self.d_gr_bins.dtype)

        # Make kernel for updating statistics
        self.update_kernel = self.make_updater_kernel(configuration, self.compute_plan)

    def make_updater_kernel(self, configuration, compute_plan, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        ghost_particles = sef.ghost_particles
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (ghost_particles - 1) // pb + 1  

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        u_id, w_id, lap_id, m_id = [configuration.sid[key] for key in ['u', 'w', 'lap', 'm']]
        
        # Prepare user-specified functions for inclusion in kernel(s)
        ptype_function = numba.njit(self.configuration.ptype_function)
        params_function = numba.njit(self.pair_potential.params_function)
        pairpotential_function = numba.njit(self.pair_potential.pairpotential_function)
        dist_sq_function = numba.njit(self.configuration.simbox.dist_sq_function)

        def update_kernel(vectors, sim_box, ptype, ghost_positions, boltzmann_factors):
            """ Calculate g(r) fresh
            Kernel configuration: [num_blocks, (pb, tp)]
        """

            #num_bins = d_gr_bins.shape[0]  # reading number of bins from size of the device array
            #min_box_dim = min(sim_box[0], sim_box[1], sim_box[2])  # max distance for rdf can 0.5*Smallest dimension
            #bin_width = (min_box_dim / 2) / num_bins  # TODO: Chose more directly!
            u = np.float32(0.0)

            global_id = cuda.grid(1)
            if global_id < ghost_particles:
                for other_global_id in range(0, num_part, 1):
                    dist_sq = dist_sq_function(vectors[r_id][other_global_id], 
                                               ghost_positions[global_id], sim_box)
                    u += dist_sq
                boltzmann_factors[global_id] += u

            return

        return cuda.jit(device=False)(update_kernel)[num_blocks, (pb,)]

    def update(self):
        """ Update the radial distribution function with the current configuration. """
        self.count += 1
        self.update_kernel(self.configuration.d_vectors,
                           self.configuration.simbox.d_data,
                           self.configuration.d_ptype,
                           self.d_ghost_positions,
                           self.d_boltzmann_factors)
        self.rdf_list.append(self.d_gr_bins.copy_to_host())
        self.d_gr_bins = cuda.to_device(self.host_array_zeros)

    def read(self):
        """ Read the radial distribution function

        Returns
        -------

        dict
            A dictionary containing the distances and the radial distribution function.
        """
        num_bins = self.rdf_list[0].shape[0]
        min_box_dim = min(self.configuration.simbox.lengths[0], self.configuration.simbox.lengths[1],
                          self.configuration.simbox.lengths[2])
        bin_width = (min_box_dim / 2) / num_bins
        rdf = np.array(self.rdf_list)

        # Normalize the g(r) lengths # Compute in setup and normalize om the fly 
        rho = self.configuration.N / np.prod(self.configuration.simbox.lengths)
        for i in range(num_bins):  # Normalize one bin/distance at a time
            r_outer = (i + 1) * bin_width
            r_inner = i * bin_width
            shell_volume = (4.0 / 3.0) * np.pi * (r_outer ** 3 - r_inner ** 3)
            expected_num = rho * shell_volume
            rdf[:, i] /= (expected_num * self.configuration.N)

        distances = np.arange(0, num_bins) * bin_width
        return {'distances': distances, 'rdf': rdf}

    def save_average(self, output_filename="rdf.dat") -> None:
        """ Save the average radial distribution function to a file

        Parameters
        ----------

        output_filename : str
            The name of the file to which the radial distribution function is saved.
        """

        rdf_dict = self.read()
        np.savetxt(output_filename, np.c_[rdf_dict['distances'], np.mean(rdf_dict['rdf'], axis=0)], header="r g(r)")
