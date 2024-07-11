import itertools

import numpy as np
import rumdpy as rp
import numba


class CalculatorStructureFactor:
    """ Calculator class for the structure factor of a system, S(q)
    The calculation is done for the q vectors pointing in the q_direction (0=x, 1=y, ...).
    The length of the q vectors is determined by the n_values list:
        q = (2*pi*n_x/L_x, 2*pi*n_x/L_x, ...)
    where n=(n_x, n_y, ...) is a D-dimensional vector of integers and L_x, L_y are the box lengths in the x and y  directions.
    The collective density rho_q is calculated as:
        rho_q = 1/sqrt(N) * sum_{n} exp(-i q.x_n)
    where x_n is the position of particle n
    The structure factor is defined as:
        S(q) = |rho_q|^2

    Initialize the structure factor calculator.
    ===========================================
    If n_vectors is None, the q vectors are generated from q_max.
    If n_vectors is not None, the q vectors are taken from n_vectors.
    The n_vectors must be a numpy array of shape (n, D)
    where n is the number of vectors and D is the number of dimensions.
    Note that either q_max or n_vectors must be specified.

    """

    def __init__(self, configuration: rp.Configuration,
#                 q_max: None | float = None, n_vectors=None, backend='parallel', ) -> None:
                 q_max = None, n_vectors=None, backend='parallel', ) -> None:
        self.update_count = 0
        self.configuration = configuration
        self.L = self.configuration.simbox.lengths.copy()
        dimension_of_space = self.configuration.D
        if n_vectors is not None:
            if q_max is not None:
                raise ValueError('Only one of q_max or n_vectors must be specified.')
            if not isinstance(n_vectors, np.ndarray):
                raise ValueError('n_vectors must be a numpy array.')
            if n_vectors.shape[1] != dimension_of_space:
                raise ValueError('n_vectors must have the same number of columns as the number of dimensions.')
            self.n_vectors = n_vectors
            self.q_vectors = np.array(2 * np.pi * self.n_vectors / self.L, dtype=np.float32)
        else:
            # n_vectors = [[0, 0, 1], [0, 0, 2], ..., [0, 1, 0], [0, 1, 1] ..., [18, 18, 18], ...]
            if q_max is None:
                raise ValueError('q_max or n_vectors must be specified.')
            n_max = int(np.ceil(q_max * max(self.L) / (2 * np.pi)))
            self.n_vectors = np.array(list(itertools.product(range(n_max), repeat=dimension_of_space)), dtype=int)
            self.n_vectors = self.n_vectors[1:]  # Remove the first vector [0, 0, 0]
            self.q_vectors = np.array(2 * np.pi * self.n_vectors / self.L, dtype=np.float32)
            # Remove q_vectors where the length is greater than q_max
            self.q_vectors = self.q_vectors[np.linalg.norm(self.q_vectors, axis=1) < q_max]
        self.q_lengths = np.linalg.norm(self.q_vectors, axis=1)
        self.list_of_rho_q = []
        self.list_of_rho_S_q = []
        self.sum_S_q = np.zeros_like(self.q_lengths)
        self.compute_rho_q = self._generate_compute_rho_q(backend)

    @staticmethod
    def _generate_compute_rho_q(backend='parallel'):
        if backend == 'parallel':
            def func(r_vec: np.ndarray, q_vec: np.ndarray):
                num_particles = r_vec.shape[0]
                number_of_q_vectors = q_vec.shape[0]
                rho_q = np.zeros(number_of_q_vectors, dtype=np.complex64)
                for i in numba.prange(number_of_q_vectors):
                    r_dot_q = np.dot(r_vec, q_vec[i])
                    rho_q[i] = np.sum(np.exp(1j * r_dot_q))*num_particles**(-1/2)
                return rho_q
            return numba.njit(parallel=True)(func)
        elif backend == 'single core':
            def func(r_vec: np.ndarray, q_vec: np.ndarray):
                N = r_vec.shape[0]
                r_dot_q = np.dot(r_vec, q_vec.T)
                rho_q = np.sum(np.exp(1j * r_dot_q), axis=0)*N**(-1/2)
                return rho_q
            return numba.njit(func)
        else:
            raise ValueError(f'Unknown backend, {backend}')

    def update(self) -> None:
        """ Update the structure factor with the current configuration. """
        if not np.allclose(self.L, self.configuration.simbox.lengths):
            raise ValueError('Box length has changed. Recreate the S(q) object.')
        this_rho_q = self.compute_rho_q(self.configuration['r'], self.q_vectors)
        self.list_of_rho_q.append(this_rho_q)
        self.list_of_rho_S_q.append(np.abs(this_rho_q)**2)
        self.sum_S_q += np.abs(this_rho_q) ** 2
        self.update_count += 1

    #def read(self, bins: int | None) -> dict:
    def read(self, bins) -> dict:
        """ Return the structure factor S(q) for the q vectors in the q_direction.
            If bins is an integer, the data is binned (ready to be plotted).
            If bins is None, the raw S(q) data is returned.
        """
        if isinstance(bins, int):
            q_bins = np.linspace(0, np.max(self.q_lengths), bins+1)
            q_binned = np.zeros(bins, dtype=np.float32)
            S_q_binned = np.zeros(bins, dtype=np.float32)
            q_vectors_in_bin = np.zeros(bins, dtype=int)
            for i in range(0, bins):
                mask = (q_bins[i] <= self.q_lengths) & (self.q_lengths < q_bins[i+1])
                if np.sum(mask) == 0:  # No q vectors in this bin
                    continue
                q_binned[i] = np.mean(self.q_lengths[mask])
                # Use self.list_of_rho_S_q
                S_q_binned[i] = np.mean(self.sum_S_q[mask] / self.update_count)
                q_vectors_in_bin[i] = np.sum(mask)
            # Remove bins with no q vectors
            mask = q_vectors_in_bin > 0
            q_binned = q_binned[mask]
            S_q_binned = S_q_binned[mask]
            q_vectors_in_bin = q_vectors_in_bin[mask]
            return {
                '|q|': q_binned,
                'S(|q|)': S_q_binned,
                'q_vectors_in_bin': q_vectors_in_bin
            }
        elif bins is None:
            # Return (un-binned) raw data
            return {
                'q': self.q_vectors,
                '|q|': self.q_lengths,
                'S(q)': self.sum_S_q/self.update_count,
                'rho_q': np.array(self.list_of_rho_q),
                'n_vectors': self.n_vectors
            }
        else:
            raise ValueError('bins must be an integer.')

    #def save_average(self, bins: int=None, output_filename: str="sq.dat") -> None:
    def save_average(self, bins=None, output_filename="sq.dat"):
        if bins is None: bins=100
        sq_dict = self.read(bins)
        np.savetxt(output_filename, np.c_[sq_dict['|q|'], sq_dict['S(|q|)']], header="|q| S(|q|)")
