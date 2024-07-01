def make_lattice(unit_cell: list, lattice_constants: list, cells: list, rho=None) -> tuple:
    """ Returns a configuration of a crystal lattice.
    The lattice is constructed by replicating the unit cell in all directions.
    The coordinates (`unit_cell`) of the unit cell are given in fractional coordinates.
    The `lattice_constants` a list of lengths of the unit cell in all directions.
    The `cells` are the number of unit cells in each direction.

    Returns a list of positions of the atoms in the lattice, and the box vector of the lattice.

    Example
    -------

    >>> import rumdpy as rp
    >>> fcc_unit_cell = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    >>> lattice_constants = [1.0, 1.0, 1.0]
    >>> cells = [8, 8, 8]
    >>> positions, box_vector = rp.tools.make_lattice(fcc_unit_cell, lattice_constants, cells)
    >>> configuration = rp.Configuration()
    >>> configuration['r'] = positions
    >>> configuration.simbox = rp.Simbox(len(box_vector), box_vector)
    """
    import numpy as np
    particles_in_unit_cell = len(unit_cell)
    spatial_dimension = len(unit_cell[0])
    number_of_cells = np.prod(cells)
    positions = np.zeros(
        shape=(particles_in_unit_cell * number_of_cells, spatial_dimension),
        dtype=np.float64,
    )
    for cell_index in range(number_of_cells):
        cell_coordinates = np.array(
            np.unravel_index(cell_index, cells), dtype=np.float64
        )
        for particle_index in range(particles_in_unit_cell):
            positions[cell_index * particles_in_unit_cell + particle_index] = (
                unit_cell[particle_index] + cell_coordinates
            )
    positions *= lattice_constants
    box_vector = np.array(lattice_constants) * np.array(cells)
    if rho is not None:
        box_volume = np.prod(box_vector)
        number_of_particles = len(positions)
        volume_per_particle = box_volume / number_of_particles
        target_volume_per_particle = 1.0 / rho
        scale_factor = (target_volume_per_particle / volume_per_particle) ** (1.0 / 3.0)
        positions *= scale_factor
        box_vector *= scale_factor
    return positions, box_vector
