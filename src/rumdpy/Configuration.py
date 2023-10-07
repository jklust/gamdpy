import numpy as np
import numba
import math
from numba import cuda

class Configuration():
    
    vid = {'r':0, 'v':1, 'f':2, 'r_ref':3}
    sid = {'u':0, 'w':1, 'lap':2, 'm':3, 'k':4, 'fsq':5}
    num_cscalars = 3 # Number of scalars to be updated by force calculator
    
    def __init__(self, N, D, simbox_data, ftype=np.float32):
        self.N = N
        self.D = D
        self.vectors = np.zeros((len(self.vid), N, D), dtype=ftype)
        self.scalars = np.zeros((N, len(self.sid)), dtype=ftype)
        self.r_im = np.zeros((N, D), dtype=np.int)
        self.ptype = np.zeros(N, dtype=np.int)
        self.ptype_function = self.make_ptype_function()
        self.simbox = simbox(D, simbox_data)
        return
    
    def set_vector(self, key, data):
        N, D = data.shape
        assert N==self.N, f'Inconsistent number of particles, {N} <> {self.N}'
        assert D==self.D, f'Inconsistent number of dimensions, {D} <> {self.D}'
        # assert key exists
        self.vectors[self.vid[key], :, :] = data
        return

    def set_scalar(self, key, data):
        N, = data.shape
        assert N==self.N, f'Inconsistent number of particles, {N} <> {self.N}'
        # assert key exists
        self.scalars[:, self.sid[key]] = data
        return
        
    def __setitem__(self, key, data):
        if key in self.vid.keys():
            self.set_vector(key, data)
        else:
            self.set_scalar(key, data) # Improve error handling if key in neither
        return 
    
    def __getitem__(self, key):
        if key in self.vid.keys():
            return self.vectors[self.vid[key]]
        return self.scalars[self.sid[key]] # Improve error handling if key in neither
    
    def copy_to_device(self):
        self.d_vectors = cuda.to_device(self.vectors)
        self.d_scalars = cuda.to_device(self.scalars)
        self.d_r_im = cuda.to_device(self.r_im)
        self.d_ptype = cuda.to_device(self.ptype)
        self.simbox.copy_to_device()
        return
    
    def make_ptype_function(self):
        def ptype_function(pid, ptype_array):
            ptype = ptype_array[pid]          # Default: read from ptype_array
            return ptype
        return ptype_function

class simbox():
    def __init__(self, D, data):
        self.D = D
        self.data = data.copy()
        self.dist_sq_dr_function, self.dist_sq_function = self.make_simbox_functions()
        return

    def copy_to_device(self):
        self.d_data = cuda.to_device(self.data)
        
    def make_simbox_functions(self):
        D = self.D
        def dist_sq_dr_function(ri, rj, sim_box, dr): # simbox could be compiled in, but not so flexible e.g for NpT
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr[k] = ri[k] - rj[k]
                box_k = sim_box[k]
                dr[k] += (-box_k if numba.float32(2.0)*dr[k]>+box_k else 
                          (+box_k if numba.float32(2.0)*dr[k]<-box_k else numba.float32(0.0))) # MIC 
                dist_sq = dist_sq + dr[k]*dr[k]
            return dist_sq

        def dist_sq_function(ri, rj, sim_box): # simbox could be compiled in, but not so flexible e.g for NpT
            dist_sq = numba.float32(0.0)
            for k in range(D):
                dr_k = ri[k] - rj[k]
                box_k = sim_box[k]
                dr_k += (-box_k if numba.float32(2.0)*dr_k>+box_k else 
                         (+box_k if numba.float32(2.0)*dr_k<-box_k else numba.float32(0.0))) # MIC 
                dist_sq = dist_sq + dr_k*dr_k
            return dist_sq   
    
        return dist_sq_dr_function, dist_sq_function
    
# Helper functions

def generate_random_velocities(N, D, T, m=1, dtype=np.float32):
    v = dtype(np.random.normal(0.0, T**0.5, (N,D)))    ###### Assuming m=1, INSERT CORRECT FORMULA ######
    for k in range(D):
        v[:,k] -= np.mean(v[:,k]) # remove drift
    T_ = np.sum(m*v**2)/(D*(N-1))
    v *= (T/T_)**0.5
    return dtype(v)


@numba.njit
def generate_fcc_positions(nx, ny, nz, rho, dtype=np.float32):
    D=3
    conf = np.zeros((nx*ny*nz*4, D), dtype=dtype)
    count = 0 
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                conf[count+0,:] = [ix+0.25, iy+0.25, iz+0.25]
                conf[count+1,:] = [ix+0.75, iy+0.75, iz+0.25]
                conf[count+2,:] = [ix+0.75, iy+0.25, iz+0.75]
                conf[count+3,:] = [ix+0.25, iy+0.75, iz+0.75]
                count += 4
    for k in range(D):
        conf[:,k] -= np.mean(conf[:,k]) # put sample in the middle of the box
    sim_box = np.array((nx, ny, nz), dtype=dtype)
    rho_initial = 4.0
    scale_factor = dtype((rho_initial/rho)**(1/D))
    
    return conf*scale_factor, sim_box*scale_factor

def params_function(i_type, j_type, params):
    result = params[i_type, j_type]             # default: read from params array
    return result

##### Simbox, so far only functions... Should include also 'put back in box'-function ############


