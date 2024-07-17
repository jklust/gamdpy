import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import \
    make_fixed_interactions  # planar interactions is an example of 'fixed' interactions


#########################################################################
######## Planar interactions (smmoth walls, gravity, electric field)
#########################################################################

def make_planar_calculator(configuration, potential_function) -> callable:
    """ Returns a function that calculates a planar inteaction for particles

    This function is used to create a planar interaction such as a smooth wall, gravity or an electric field.

    """

    D = configuration.D
    dist_sq_dr_function = numba.njit(configuration.simbox.dist_sq_dr_function)
    dist_sq_function = numba.njit(configuration.simbox.dist_sq_function)

    # Unpack indices for vectors and scalars
    #for key in configuration.vid:
    #    exec(f'{key}_id = {configuration.vid[key]}', globals())
    for col in configuration.vectors.column_names:
        exec(f'{col}_id = {configuration.vectors.indices[col]}', globals())
    for key in configuration.sid:
        exec(f'{key}_id = {configuration.sid[key]}', globals())

    def planar_calculator(vectors, scalars, ptype, sim_box, indices, values):
        particle = indices[0]
        interaction_type = indices[1]
        point = values[interaction_type][0:D]  # Point in wall
        normal_vector = values[interaction_type][D:2 * D]  # Normal vector defining plane of wall

        # Calculating full D-dim displacement vector to avoid worrying about new sim_box types in future
        dr = cuda.local.array(shape=D, dtype=numba.float32)
        dist_sq = dist_sq_dr_function(point, vectors[r_id][particle], sim_box, dr)
        dist = numba.float32(0.0)
        for k in range(D):
            dist += dr[k] * normal_vector[k]
        if dist < values[interaction_type][-1]:  # Last index is the cut-off
            u, s, umm = potential_function(abs(dist),
                                           values[interaction_type][2 * D:])  # abs: potential symmetric around wall

            for k in range(D):
                cuda.atomic.add(vectors, (f_id, particle, k), -normal_vector[k] * dist * s)  # Force
            cuda.atomic.add(scalars, (particle, w_id), dist ** 2 * s)  # Virial
            cuda.atomic.add(scalars, (particle, u_id), u)  # Potential enerrgy
            lap = numba.float32(1 - D) * s + umm  # Laplacian
            cuda.atomic.add(scalars, (particle, lap_id), lap)

        return

    return planar_calculator


def setup_planar_interactions(configuration, potential_function, potential_params_list, particles_list, point_list,
                              normal_vector_list, compute_plan, verbose=True) -> dict:
    """ Returns a dictionary with planar interactions """
    D = configuration.D
    num_types = len(potential_params_list)
    assert len(particles_list) == num_types
    assert len(point_list) == num_types
    assert len(normal_vector_list) == num_types

    total_number_indices = 0
    for particles in particles_list:
        total_number_indices += particles.shape[0]

    if verbose:
        print(
            f'Setting up planar interactions: {num_types} types, {total_number_indices} particle-plane interactions in total.')

    indices = np.zeros((total_number_indices, 2), dtype=np.int32)
    params = np.zeros((num_types, 2 * D + len(potential_params_list[0])), dtype=np.float32)

    start_index = 0
    for interaction_type in range(num_types):
        next_start_index = start_index + len(particles_list[interaction_type])
        indices[start_index:next_start_index, 0] = particles_list[interaction_type]
        indices[start_index:next_start_index, 1] = interaction_type
        start_index = next_start_index

        params[interaction_type, 0:D] = point_list[interaction_type]
        params[interaction_type, D:2 * D] = normal_vector_list[interaction_type]  # Normalize it!
        params[interaction_type, 2 * D:] = potential_params_list[interaction_type]

    calculator = make_planar_calculator(configuration, potential_function)
    interactions = make_fixed_interactions(configuration, calculator, compute_plan, verbose=False)
    d_indices = cuda.to_device(indices)
    d_params = cuda.to_device(params)
    interaction_params = (d_indices, d_params)

    return {'interactions': interactions, 'interaction_params': interaction_params}
