def make_lattice(unit_cell: dict, cells: list = None, rho=None) -> tuple:
    """ Returns a configuration of a crystal lattice.
    The lattice is constructed by replicating the unit cell in all directions.
    The `unit_cell` is a dictonary with `fractional_coordinates` for particles, and
    the `lattice_constants` as a list of lengths of the unit cell in all directions.
    The `cells` are the number of unit cells in each direction.

    Returns a list of positions of the atoms in the lattice, and the box vector of the lattice.

    Example
    -------

    >>> import rumdpy as rp
    >>> positions, box_vector = rp.tools.make_lattice(rp.FCC, cells=[8, 8, 8], rho=1.0)
    >>> configuration = rp.Configuration()
    >>> configuration['r'] = positions
    >>> configuration.simbox = rp.Simbox(configuration.D, box_vector)
    """
    import numpy as np
    pos = unit_cell["fractional_coordinates"]
    lat = unit_cell["lattice_constants"]
    particles_in_unit_cell = len(pos)
    spatial_dimension = len(pos[0])
    if cells is None:
        cells = [1] * spatial_dimension
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
                pos[particle_index] + cell_coordinates
            )
    positions *= lat
    box_vector = np.array(lat) * np.array(cells)
    if rho is not None:
        box_volume = np.prod(box_vector)
        number_of_particles = len(positions)
        volume_per_particle = box_volume / number_of_particles
        target_volume_per_particle = 1.0 / rho
        scale_factor = (target_volume_per_particle / volume_per_particle) ** (1.0 / 3.0)
        positions *= scale_factor
        box_vector *= scale_factor
    return positions, box_vector
