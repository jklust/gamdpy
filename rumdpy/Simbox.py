#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:41:24 2024

@author: nbailey
"""

import numpy as np
import numba
from numba import cuda

class Simbox():
    """ Simulation box class 

    Example
    -------

    >>> import rumdpy as rp
    >>> import numpy as np
    >>> simbox = rp.Simbox(D=3, lengths=np.array([3,4,5]))
    >>> assert list(simbox.lengths) == list(np.array([3,4,5]))
    """
    def __init__(self, D, lengths):
        self.D = D
        self.lengths = np.array(lengths, dtype=np.float32) # ensure single precision
        self.len_sim_box_data = D # not true for other Simbox classes
        self.dist_sq_dr_function, self.dist_sq_function, self.apply_PBC, self.volume, self.dist_moved_sq_function, self.dist_moved_exceeds_limit_function = self.make_simbox_functions()
        return

    def make_device_copy(self):
        """ Creates a new device copy of the simbox data and returns it to the caller.
        To be used by neighbor list for recording the box state at time of last rebuild"""
        return cuda.to_device(self.lengths)

    def copy_to_device(self):
        self.d_data = cuda.to_device(self.lengths)

    def copy_to_host(self):
        self.lengths = self.d_data.copy_to_host()

    def make_simbox_functions(self):
        D = self.D

        def dist_sq_dr_function(ri, rj, sim_box, dr):  
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr[k] = ri[k] - rj[k]
                box_k = sim_box[k]
                dr[k] += (-box_k if numba.float32(2.0) * dr[k] > +box_k else
                          (+box_k if numba.float32(2.0) * dr[k] < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr[k] * dr[k]
            return dist_sq

        def dist_sq_function(ri, rj, sim_box):  
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr_k = ri[k] - rj[k]
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr_k * dr_k
            return dist_sq

        def apply_PBC(r, image, sim_box):
            for k in range(D):
                if r[k] * numba.float32(2.0) > +sim_box[k]:
                    r[k] -= sim_box[k]
                    image[k] += 1
                if r[k] * numba.float32(2.0) < -sim_box[k]:
                    r[k] += sim_box[k]
                    image[k] -= 1

        def apply_PBC_dimension(r, image, sim_box, dimension):
            if r[dimension] * numba.float32(2.0) > +sim_box[dimension]:
                r[dimension] -= sim_box[dimension]
                image[dimension] += 1
            if r[dimension] * numba.float32(2.0) < -sim_box[dimension]:
                r[dimension] += sim_box[dimension]
                image[dimension] -= 1

        def volume(sim_box):
            vol = sim_box[0]
            for i in range(1,D):
                vol *= sim_box[i]
            return vol

        def dist_moved_sq_function(r_current, r_last, sim_box, sim_box_last):
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr_k * dr_k

            return dist_sq


        def dist_moved_exceeds_limit_function(r_current, r_last, sim_box, sim_box_last, skin, cut):
            """ Parameters sim_box_last and cut are not used here, but is needed for the Lees-Edwards type of Simbox"""
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr_k * dr_k

            return dist_sq > skin*skin*numba.float32(0.25)

        return dist_sq_dr_function, dist_sq_function,  apply_PBC, volume, dist_moved_sq_function, dist_moved_exceeds_limit_function



class Simbox_LeesEdwards(Simbox):
    """ Simulation box class with LeesEdwards bondary conditions

    Example
    -------

    >>> import rumdpy as rp
    >>> import numpy as np
    >>> simbox = rp.Simbox_LeesEdwards(D=3, lengths=np.array([3,4,5]), box_shift=1.0)
    Simbox_LeesEdwards, box_shift= 1.0
    >>> assert list(simbox.lengths) == list(np.array([3,4,5]))

    """
    def __init__(self, D, lengths, box_shift=0.):
        if D < 2:
            raise ValueError("Cannot use Simbox_LeesEdwards with dimension smaller than 2")
        Simbox.__init__(self, D, lengths)
        self.box_shift = box_shift
        self.box_shift_image = 0.
        self.len_sim_box_data = D+2 # for saving purposes; we don't include the last four items (last_box_shift etc) here
        print('Simbox_LeesEdwards, box_shift=', box_shift)

        # have already called base class Simbox.make_simbox_functions, and can
        # re-use the volume so this version only has to override the first
        # three and dist_moved_sq_function
        self.dist_sq_dr_function, self.dist_sq_function, self.apply_PBC, self.update_box_shift, self.dist_moved_sq_function, self.dist_moved_exceeds_limit_function = self.make_simbox_functions_LE()

        return

    def copy_to_device(self):
        # Here it assumed this is being done for the first time

        D = self.D
        # will become a D+2-length array when we start using simbox_last_rebuild
        data_array = np.zeros(D+6, dtype=np.float32) # extra entries are: box_shift, box_shift_image, last box_shift, 
        # last_box_shift_image (ie last time NB list was built), strain change since NB list was built, correction to skin due to strain change
        data_array[:D] = self.lengths[:]
        data_array[D] = self.box_shift
        data_array[D+1] = self.box_shift_image
        self.d_data = cuda.to_device(data_array)

    def make_device_copy(self):
        """ Creates a new device copy of the simbox data and returns it to the caller.
        To be used by neighbor list for recording the box state at time of last rebuild"""
        #host_copy = self.d_data.copy_to_host()
        D = self.D
        host_copy = np.zeros(D+2)
        host_copy[:D] = self.lengths[:]
        host_copy[D] = self.box_shift
        host_copy[D+1] = self.box_shift_image
        return cuda.to_device(host_copy)

    def copy_to_host(self):
        D = self.D
        box_data =  self.d_data.copy_to_host()
        self.lengths = box_data[:D].copy()
        self.box_shift = box_data[D]
        self.boxshift_image = box_data[D+1]
        # don't need last_box_shift etc on the host except maybe occasionally for debugging?

    def make_simbox_functions_LE(self):
        D = self.D

        def dist_sq_dr_function(ri, rj, sim_box, dr):  
            box_shift = sim_box[D]
            for k in range(D):
                dr[k] = ri[k] - rj[k]

            dist_sq = numba.float32(0.0)
            box_1 = sim_box[1]
            dr[0] += (-box_shift if numba.float32(2.0) * dr[1] > +box_1 else
                      (+box_shift if numba.float32(2.0) * dr[1] < -box_1 else
                        numba.float32(0.0)))

            for k in range(D):
                box_k = sim_box[k]
                dr[k] += (-box_k if numba.float32(2.0) * dr[k] > +box_k else
                          (+box_k if numba.float32(2.0) * dr[k] < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr[k] * dr[k]
            return dist_sq

        def dist_sq_function(ri, rj, sim_box):  
            box_shift = sim_box[D]
            dist_sq = numba.float32(0.0)

            # first shift the x-component depending on whether the y-component is wrapped
            dr1 = ri[1] - rj[1]
            box_1 = sim_box[1]
            x_shift = (-box_shift if numba.float32(2.0) * dr1 > box_1 else
                      (+box_shift if numba.float32(2.0) * dr1 < -box_1 else
                        numba.float32(0.0)))
            # then wrap as usual for all components
            for k in range(D):
                dr_k = ri[k] - rj[k]
                if k == 0:
                    dr_k += x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
                dist_sq = dist_sq + dr_k * dr_k
            return dist_sq

        def apply_PBC(r, image, sim_box):

            # first shift the x-component depending on whether the y-component is outside the box
            box_shift, bs_image = sim_box[D], int(sim_box[D+1])
            box1_half = sim_box[1] * numba.float32(0.5)
            if r[1] > + box1_half:
                r[0] -= box_shift
                image[0] -= bs_image
            if r[1] < -box1_half:
                r[0] += box_shift
                image[0] += bs_image
            # then put everything back in the box as usual
            for k in range(D):
                if r[k] * numba.float32(2.0) > +sim_box[k]:
                    r[k] -= sim_box[k]
                    image[k] += 1
                if r[k] * numba.float32(2.0) < -sim_box[k]:
                    r[k] += sim_box[k]
                    image[k] -= 1


        def update_box_shift(sim_box, shift):
            # carry out the addition in double precision
            sim_box[D] = numba.float32(sim_box[D] + numba.float64(shift))
            Lx = sim_box[0]
            Lx_half = Lx*numba.float32(0.5)
            if sim_box[D] > +Lx_half:
                sim_box[D] -= Lx
                sim_box[D+1] += 1
            if sim_box[D] < -Lx_half:
                sim_box[D] += Lx
                sim_box[D+1] -= 1


        def dist_moved_sq_function(r_current, r_last, sim_box, sim_box_last):
            zero = numba.float32(0.)
            half = numba.float32(0.5)
            one = numba.float32(1.0)
            box_shift = sim_box[D]
            dist_moved_sq = zero


            strain_change = sim_box[D] - sim_box_last[D] # change in box-shift
            strain_change += (sim_box[D+1] - sim_box_last[D+1]) * sim_box[0] # add contribution from box_shift_image
            strain_change /= sim_box[1] # convert to (xy) strain
            #strain_change = sim_box[D+4]

            # we will shift the x-component when the y-component is 'wrapped'
            dr1 = r_current[1] - r_last[1]
            box_1 = sim_box[1]
            y_wrap = (one if dr1 > half*box_1 else
                      -one if dr1 < -half*box_1 else zero)

            x_shift = y_wrap * box_shift + (r_current[1] -
                                            y_wrap*box_1) * strain_change
            # see the expression in Chatoraj Ph.D. thesis. Adjusted here to
            # take into account BC wrapping (otherwise would use the images
            # ie unwrapped positions)

            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                if k == 0:
                    dr_k -= x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))
                dist_moved_sq = dist_moved_sq + dr_k * dr_k


            return dist_moved_sq

        def dist_moved_exceeds_limit_function(r_current, r_last, sim_box, sim_box_last, skin, cut):
            zero = numba.float32(0.)
            half = numba.float32(0.5)
            one = numba.float32(1.0)
            box_shift = sim_box[D]
            dist_moved_sq = zero


            strain_change = sim_box[D] - sim_box_last[D] # change in box-shift
            strain_change += (sim_box[D+1] - sim_box_last[D+1]) * sim_box[0] # add contribution from box_shift_image
            strain_change /= sim_box[1] # convert to (xy) strain

            # we will shift the x-component when the y-component is 'wrapped'
            dr1 = r_current[1] - r_last[1]
            box_1 = sim_box[1]
            y_wrap = (one if dr1 > half*box_1 else
                      -one if dr1 < -half*box_1 else zero)

            x_shift = y_wrap * box_shift + (r_current[1] -
                                            y_wrap*box_1) * strain_change
            # see the expression in Chatoraj Ph.D. thesis. Adjusted here to
            # take into account BC wrapping (otherwise would use the images
            # ie unwrapped positions)

            for k in range(D):
                dr_k = r_current[k] - r_last[k]
                if k == 0:
                    dr_k -= x_shift
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                         (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))
                dist_moved_sq = dist_moved_sq + dr_k * dr_k

            skin_corrected = skin - abs(strain_change)*cut
            if skin_corrected < zero:
                skin_corrected = zero

            return dist_moved_sq > skin_corrected*skin_corrected*numba.float32(0.25)

        return dist_sq_dr_function, dist_sq_function,  apply_PBC, update_box_shift, dist_moved_sq_function, dist_moved_exceeds_limit_function
