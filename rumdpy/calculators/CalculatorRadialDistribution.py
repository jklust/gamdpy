import numpy as np
import rumdpy as rp
import numba
import math
from numba import cuda


#############################################################
#### Radial Distribution Function  
#############################################################

class CalculatorRadialDistribution():

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
        D = configuration.D
        num_part = configuration.N
        pb = compute_plan['pb']
        tp = compute_plan['tp']
        UtilizeNIII = compute_plan['UtilizeNIII']
        gridsync = compute_plan['gridsync']
        num_blocks = (num_part - 1) // pb + 1

        # Unpack indices for vectors and scalars
        # look up the indices needed instead (and throw error if not there)
        for col in configuration.vectors.column_names:
            exec(f'{col}_id = {configuration.vectors.indices[col]}', globals())

        for key in configuration.sid:
            exec(f'{key}_id = {configuration.sid[key]}', globals())

        # Prepare user-specified functions for inclusion in kernel(s)
        ptype_function = numba.njit(configuration.ptype_function)
        #params_function = numba.njit(pair_potential.params_function)
        dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

        def rdf_calculator_full(vectors, sim_box, ptype, d_gr_bins):
            """ Calculate g(r) fresh
            Kernel configuration: [num_blocks, (pb, tp)]
        """

            num_bins = d_gr_bins.shape[0]  # reading number of bins from size of the device array
            min_box_dim = min(sim_box[0], sim_box[1], sim_box[2])  # max distance for rdf can 0.5*Smallest dimension
            bin_width = (min_box_dim / 2) / num_bins  # TODO: Chose more directly!

            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block * pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_part:
                for i in range(0, num_part, pb * tp):
                    for j in range(pb):
                        other_global_id = j + i + my_t * pb
                        if other_global_id != global_id and other_global_id < num_part:
                            dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id],
                                                       sim_box)

                            # Calculate g(r)
                            if dist_sq < (min_box_dim / 2) ** 2:
                                dist = math.sqrt(dist_sq)
                                if dist < min_box_dim / 2:
                                    bin_index = int(dist / bin_width)
                                    cuda.atomic.add(d_gr_bins, bin_index, 1)

            return

        return cuda.jit(device=0)(rdf_calculator_full)[num_blocks, (pb, tp)]

    def update(self):
        self.count += 1
        self.update_kernel(self.configuration.d_vectors,
                           self.configuration.simbox.d_data,
                           self.configuration.d_ptype,
                           self.d_gr_bins)
        self.rdf_list.append(self.d_gr_bins.copy_to_host())
        self.d_gr_bins = cuda.to_device(self.host_array_zeros)

    def read(self):
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

    def save_average(self, output_filename="rdf.dat"):
        rdf_dict = self.read()
        np.savetxt(output_filename, np.c_[rdf_dict['distances'], np.mean(rdf_dict['rdf'], axis=0)])

