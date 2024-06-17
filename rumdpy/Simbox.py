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
    def __init__(self, D, lengths):
        self.D = D
        self.lengths = np.array(lengths, dtype=np.float32) # ensure single precision
        self.dist_sq_dr_function, self.dist_sq_function, self.apply_PBC, self.volume = self.make_simbox_functions()
        return

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

        return dist_sq_dr_function, dist_sq_function,  apply_PBC, volume
    
    
    
class Simbox_LeesEdwards(Simbox):
    def __init__(self, D, lengths, box_shift=0.):
        if D < 2:
            raise ValueError("Cannot use Simbox_LeesEdwards with dimension smaller than 2")
        Simbox.__init__(self, D, lengths)
        self.box_shift = box_shift
        print('Simbox_LeesEdwards, box_shift=', box_shift)

        # have already called base class Simox.make_simbox_functions, and can re-use the volume
        # so this version only has to override the first three
        self.dist_sq_dr_function, self.dist_sq_function, self.apply_PBC, self.update_box_shift = self.make_simbox_functions_LE()
        

        return

    def copy_to_device(self):
        D = self.D
        data_array = np.zeros(D+1, dtype=np.float32)
        data_array[:D] = self.lengths[:]
        data_array[D] = self.box_shift
        self.d_data = cuda.to_device(data_array)
        #self.d_data = cuda.to_device(np.append(self.lengths, self.box_shift))

    def copy_to_host(self):
        D = self.D
        box_data =  self.d_data.copy_to_host()
        self.lengths = box_data[:D].copy()
        self.box_shift = box_data[D]
        
        
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
            box_shift = sim_box[D]
            box1_half = sim_box[1] * numba.float32(0.5)
            if r[1] > + box1_half:
                r[0] -= box_shift
            if r[1] < -box1_half:
                r[0] += box_shift
            # then put everything back in the box as usual
            for k in range(D):
                if r[k] * numba.float32(2.0) > +sim_box[k]:
                    r[k] -= sim_box[k]
                    image[k] += 1
                if r[k] * numba.float32(2.0) < -sim_box[k]:
                    r[k] += sim_box[k]
                    image[k] -= 1
    
    
        def update_box_shift(sim_box, shift):
            global_id, my_t = cuda.grid(2)

            if global_id == 0 and my_t == 0:
                sim_box[D] += shift
                Lx = sim_box[0]
                Lx_half = Lx*numba.float32(0.5)
                if sim_box[D] > +Lx_half:
                    sim_box[D] -= Lx
                if sim_box[D] < -Lx_half:
                    sim_box[D] += Lx

        return dist_sq_dr_function, dist_sq_function,  apply_PBC, update_box_shift
