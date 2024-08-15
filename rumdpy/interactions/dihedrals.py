import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions

class Dihedrals(): 

    def __init__(self, indices, parameters):
        
        self.indices = np.array(indices, dtype=np.int32) 
        self.params = np.array(parameters, dtype=np.float32)


    def get_params(self, configuration, compute_plan, verbose=False):

        self.d_indices = cuda.to_device(self.indices)
        self.d_params = cuda.to_device(self.params)
        
        return (self.d_indices, self.d_params)

    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        u_id = configuration.sid['u']

        dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    
        def dihedral_calculator(vectors, scalars, ptype, sim_box, indices, values):
           
            p = cuda.local.array(shape=6,dtype=numba.float32)
            numbers = cuda.local.array(shape=6,dtype=numba.float32)

            for n in range(6):
                p[n] = values[indices[4]][n]
                numbers[n] = n

            dr_1 = cuda.local.array(shape=D,dtype=numba.float32)
            dr_2 = cuda.local.array(shape=D,dtype=numba.float32)
            dr_3 = cuda.local.array(shape=D,dtype=numba.float32)
            
            dist_sq_dr_function(vectors[r_id][indices[1]], vectors[r_id][indices[0]], sim_box, dr_1)
            dist_sq_dr_function(vectors[r_id][indices[2]], vectors[r_id][indices[1]], sim_box, dr_2)
            dist_sq_dr_function(vectors[r_id][indices[3]], vectors[r_id][indices[2]], sim_box, dr_3)

            c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
            c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
            c13 = dr_1[0]*dr_3[0] + dr_1[1]*dr_3[1] + dr_1[2]*dr_3[2]
            c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]
            c23 = dr_2[0]*dr_3[0] + dr_2[1]*dr_3[1] + dr_2[2]*dr_3[2]
            c33 = dr_3[0]*dr_3[0] + dr_3[1]*dr_3[1] + dr_3[2]*dr_3[2]

            cA = c13*c22 - c12*c23
            cB1 = c11*c22 - c12*c12
            cB2 = c22*c33 - c23*c23
            cD = math.sqrt(cB1*cB2)
            cc = cA/cD


            f = -(p[1]+(numbers[2]*p[2]+(numbers[3]*p[3]+(numbers[4]*p[4]+numbers[5]*p[5]*cc)*cc)*cc)*cc)
            t1 = cA
            t2 = c11*c23 - c12*c13
            t3 = -cB1
            t4 = cB2
            t5 = c13*c23 - c12*c33
            t6 = -cA
            cR1 = c12/c22
            cR2 = c23/c22

            for k in range(3):
                f1 = f*c22*(t1*dr_1[k] + t2*dr_2[k] + t3*dr_3[k])/(cD*cB1)
                f2 = f*c22*(t4*dr_1[k] + t5*dr_2[k] + t6*dr_3[k])/(cD*cB2)

                cuda.atomic.add(vectors, (f_id, indices[0], k), f1)      # Force
                cuda.atomic.add(vectors, (f_id, indices[1], k), -(1.0 + cR1)*f1 + cR2*f2)
                cuda.atomic.add(vectors, (f_id, indices[2], k), cR1*f1 - (1.0 + cR2)*f2)
                cuda.atomic.add(vectors, (f_id, indices[3], k), f2)

            u = p[0]+(p[1]+(p[2]+(p[3]+(p[4]+p[5]*cc)*cc)*cc)*cc)*cc           
            u_per_part = numba.float32(0.25)*u    

            for n in range(4):
                cuda.atomic.add(scalars, (indices[n], u_id), u_per_part) 

            return
        
        return make_fixed_interactions(configuration, dihedral_calculator, compute_plan, verbose=False)
    '''
    def get_exclusions(self, configuration, max_number_exclusions=20):
            
        exclusions = np.zeros( (configuration.N, max_number_exclusions+1), dtype=np.int32 ) 
        
        nangles = len(self.indices)
        for n in range(nangles):
            pidx = self.indices[n][:3]
            for k in range(3):

                offset = exclusions[pidx[k]][-1]
                if offset > max_number_exclusions-2:
                    raise ValueError("Number of max. exclusion breached")

                if k==0:
                    if angles_entry_not_exists(pidx[1], exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[1]
                        offset += 1
                    if angles_entry_not_exists(pidx[2], exclusions[pidx[k]],offset): 
                        exclusions[pidx[k]][offset] = pidx[2]
                        offset += 1
                elif k==1:
                    if angles_entry_not_exists(pidx[0], exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[0]
                        offset += 1
                    if angles_entry_not_exists(pidx[2], exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[2]
                        offset += 1
                else:
                    if angles_entry_not_exists(pidx[0], exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[0]
                        offset += 1 
                    if angles_entry_not_exists(pidx[1],exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[1]
                        offset += 1

                exclusions[pidx[k]][-1] = offset

        return exclusions  
                 
    def get_angle(self, angle_idx, configuration):
        
        pidx = self.indices[angle_idx][:3]

        r1 = configuration['r'][pidx[0]]
        r2 = configuration['r'][pidx[1]]
        r3 = configuration['r'][pidx[2]]
        
        dr_1 = angles_get_dist_vector(r1, r2, configuration.simbox)
        dr_2 = angles_get_dist_vector(r3, r2, configuration.simbox)

        c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
        c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
        c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]

        cD = math.sqrt(c11*c22)
        cc = c12/cD 

        angle = math.acos(cc)
        
        return angle
 
# Helpers 
def angles_get_dist_vector(ri, rj, simbox):
    dr = np.zeros(3)
    for k in range(simbox.D): 
        dr[k] = ri[k] - rj[k]
        box_k = simbox.lengths[k]
        #PP
        dr[k] += (-box_k if 2.0*dr[k] > +box_k else (+box_k if 2.0*dr[k] < -box_k else 0.0)) 

    return dr

def angles_entry_not_exists(idx, exclusion_list, nentries):

    for n in range(nentries):
        if exclusion_list[n]==idx:
            return False

    return True
    '''

