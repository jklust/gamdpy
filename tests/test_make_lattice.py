import numpy as np

import rumdpy as rp


def test_fcc_lattice(verbose=False, plot=False):
    fcc_unit_cell = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    lattice_constants = [1.0, 1.0, 1.0]
    cells = [2, 2, 2]
    positions, box_vector = rp.tools.make_lattice(fcc_unit_cell, lattice_constants, cells)
    configuration = rp.Configuration()
    configuration['r'] = positions
    configuration.simbox = rp.Simbox(len(box_vector), box_vector)
    assert configuration['r'].shape == (len(fcc_unit_cell) * np.prod(cells), len(fcc_unit_cell[0]))
    assert configuration.simbox.lengths.shape == (len(fcc_unit_cell[0]),)
    expected_positions = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [1.0, 1.0, 0.0], [1.5, 1.5, 0.0], [1.5, 1.0, 0.5], [1.0, 1.5, 0.5],
        [1.0, 0.0, 1.0], [1.5, 0.5, 1.0], [1.5, 0.0, 1.5], [1.0, 0.5, 1.5],
        [0.0, 1.0, 1.0], [0.5, 1.5, 1.0], [0.5, 1.0, 1.5], [0.0, 1.5, 1.5]])
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    if verbose:
        print("positions:", configuration['r'])
        print("box_vector:", configuration.simbox.lengths)
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(configuration['r'][:, 0], configuration['r'][:, 1], configuration['r'][:, 2])
        plt.show()


def test_bcc_lattice(verbose=False, plot=False):
    bcc_unit_cell = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    lattice_constants = [1.0, 1.0, 1.0]
    cells = [2, 2, 2]
    positions, box_vector = rp.tools.make_lattice(bcc_unit_cell, lattice_constants, cells)
    configuration = rp.Configuration()
    configuration['r'] = positions
    configuration.simbox = rp.Simbox(len(box_vector), box_vector)
    assert configuration['r'].shape == (len(bcc_unit_cell) * np.prod(cells), len(bcc_unit_cell[0]))
    assert configuration.simbox.lengths.shape == (len(bcc_unit_cell[0]),)
    expected_number_of_particles = 16
    assert configuration['r'].shape[0] == expected_number_of_particles
    expected_positions = np.array([[0., 0., 0.],
                                   [0.5, 0.5, 0.5],
                                   [0., 0., 1.],
                                   [0.5, 0.5, 1.5],
                                   [0., 1., 0.],
                                   [0.5, 1.5, 0.5],
                                   [0., 1., 1.],
                                   [0.5, 1.5, 1.5],
                                   [1., 0., 0.],
                                   [1.5, 0.5, 0.5],
                                   [1., 0., 1.],
                                   [1.5, 0.5, 1.5],
                                   [1., 1., 0.],
                                   [1.5, 1.5, 0.5],
                                   [1., 1., 1.],
                                   [1.5, 1.5, 1.5]])
    assert np.allclose(configuration['r'], expected_positions)
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    assert np.allclose(configuration.simbox.lengths, expected_box_vector)

    if verbose:
        print("positions:", configuration['r'])
        print("box_vector:", configuration.simbox.lengths)
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(configuration['r'][:, 0], configuration['r'][:, 1], configuration['r'][:, 2])
        plt.show()


def main():
    test_fcc_lattice(verbose=True, plot=True)
    test_bcc_lattice(verbose=True, plot=True)


if __name__ == "__main__":
    main()
