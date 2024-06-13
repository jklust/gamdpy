#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:41:24 2024

@author: nbailey
"""

import numba
from numba import cuda

class Simbox():
    def __init__(self, D, lengths):
        self.D = D
        self.lengths = lengths.copy()
        self.dist_sq_dr_function, self.dist_sq_function, self.apply_PBC_dimension, self.volume = self.make_simbox_functions()
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

        return dist_sq_dr_function, dist_sq_function,  apply_PBC_dimension, volume
