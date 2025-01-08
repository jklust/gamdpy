import math

import numba
import numpy as np
from numba import cuda

from .make_fixed_interactions import make_fixed_interactions  # tether is an example of 'fixed' interactions

# Abstract Base Class and type annotation
from .interaction import Interaction
from rumdpy import Configuration

class Relaxtemp(Interaction):
    """ Thermostat using simple kinetic temperature relaxation of each particles. 
 
        Parameters
        ----------
        (a) A list of particle indices to be thermostated and associated list of relaxation time 
        (b) A list of particle types to be thermostated and associated list of relaxation times

        See examples/poiseuille.py
    """
    
    def __init__(self):
        self.indices_set = False

    
    def set_relaxation_from_lists(self, particle_indices,  temperature, relax_times):

        ntau, npart, ntemp = len(tau), len(pindices), len(temperature)

        if ntau != npart or ntau != ntemp or npart != ntemp:
            raise ValueError(
                "Each particle must have exactly one relax time and temperature - arrays must be same length")

        indices, params = [], []
        for n in range(npart):
            indices.append([n, particle_indices[n]])
            params.append([temperature(n), relax_times[n]])

        self.relax_params = np.array(params, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)

        self.indices_set = True

    
    def set_relaxation_from_types(self, particle_types, temperature, relax_times, configuration): 
        
        ntypes, ntau, ntemp = len(particle_types), len(relax_times), len(temperature)

        if ntypes != ntau or ntypes != ntemp or ntemp != ntau:
            raise ValueError("Each type must have exactly one relax time - arrays must be same length")
       
        indices, params = [], []
        counter = 0
        for n in range(configuration.N):
            for m in range(ntypes):
                if configuration.ptype[n] == particle_types[m]:
                    indices.append([counter, n])
                    params.append([temperature[m], relax_times[m]])
                    counter = counter + 1
                    break

        self.relax_params = np.array(params, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)

        self.indices_set = True

    def get_params(self, configuration: Configuration, compute_plan: dict, verbose=False) -> tuple:

        if self.indices_set == False:
            raise ValueError("Indices not defined")


        self.d_pindices = cuda.to_device(self.indices)
        self.d_relax_params = cuda.to_device(self.relax_params);

        return (self.d_pindices, self.d_relax_params)

    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']]
        num_blocks = (N - 1) // pb + 1

        # Not sure if compute_flags is relevant here?? NB, Nov 2024

        # Get indices values (instead of dictonary entries) 
        v_id = configuration.vectors.indices['v']
        m_id = configuration.sid['m']

        def relaxtemp_calculator(vectors, scalars, ptype, sim_box, indices, values):
            v = vectors[v_id][indices[1]]
            m = scalars[indices[1]][m_id]
            Tdesired = values[indices[0]][0]
            tau = values[indices[0]][1]

            Tparticle = m / numba.float32(3.0) * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

            one = numba.float32(1.0)
            fac = math.sqrt(one + tau * (Tdesired / Tparticle - one))

            for k in range(D):
                v[k] = v[k] * fac

            return

        return make_fixed_interactions(configuration, relaxtemp_calculator, compute_plan, verbose=False)
