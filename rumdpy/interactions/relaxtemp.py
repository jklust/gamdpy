import math

import numba
import numpy as np
from numba import cuda

from .make_fixed_interactions import make_fixed_interactions  # tether is an example of 'fixed' interactions


class Relaxtemp:
    """ Relaxation of temperature of particles. """

    def __init__(self, tau, temperature, configuration, pindices=None, ptypes=None, verbose=False):

        indices_array, relax_params = [], []

        if pindices is None:

            ntypes, ntau, ntemp = len(ptypes), len(tau), len(temperature)

            if ntypes != ntau or ntypes != ntemp or ntemp != ntau:
                raise ValueError("Each type must have exactly one relax time - arrays must be same length")

            counter = 0
            for n in range(configuration.N):
                for m in range(ntypes):
                    if configuration.ptype[n] == ptypes:
                        indices_array.append([counter, n])
                        relax_params.append([temperature[m], tau[m]])
                        counter = counter + 1
                        break

        elif ptypes is None:

            ntau, npart, ntemp = len(tau), len(pindices), len(temperature)

            if ntau != npart or ntau != ntemp or npart != ntemp:
                raise ValueError(
                    "Each particle must have exactly one relax time and temperature - arrays must be same length")

            for n in range(npart):
                indices.append([n, pindices[n]])
                tether_params.append([temperature(n), tau[n]])

        else:
            raise ValueError("Incorrect number of arguments to constructor")

        self.relax_params = np.array(relax_params, dtype=np.float32)
        self.indices_array = np.array(indices_array, dtype=np.int32)

        if verbose:
            print(f"{self.relax_params} \n {self.indices_array}")

    def get_params(self, configuration, compute_plan, verbose=False):

        self.d_pindices = cuda.to_device(self.indices_array)
        self.d_relax_params = cuda.to_device(self.relax_params);

        return (self.d_pindices, self.d_relax_params)

    def get_kernel(self, configuration, compute_plan, compute_stresses=False, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']]
        num_blocks = (N - 1) // pb + 1

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
