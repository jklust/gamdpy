import numpy as np

import rumdpy as rp

EXPECTED_FCC_POSITIONS = np.array(
[
    [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
    [0, 1, 0], [0.5, 1.5, 0], [0.5, 1, 0.5], [0, 1.5, 0.5],
    [1, 0, 0], [1.5, 0.5, 0], [1.5, 0, 0.5], [1, 0.5, 0.5],
    [1, 1, 0], [1.5, 1.5, 0], [1.5, 1, 0.5], [1, 1.5, 0.5],
    [0, 0, 1], [0.5, 0.5, 1], [0.5, 0, 1.5], [0, 0.5, 1.5],
    [0, 1, 1], [0.5, 1.5, 1], [0.5, 1, 1.5], [0, 1.5, 1.5],
    [1, 0, 1], [1.5, 0.5, 1], [1.5, 0, 1.5], [1, 0.5, 1.5],
    [1, 1, 1], [1.5, 1.5, 1], [1.5, 1, 1.5], [1, 1.5, 1.5]
]
)

def test_fcc_lattice(verbose=False, plot=False):
    cells = [2, 2, 2]
    positions, box_vector = rp.tools.make_lattice(rp.unit_cells.FCC, cells)
    configuration = rp.Configuration()
    configuration['r'] = positions
    configuration.simbox = rp.Simbox(configuration.D, box_vector)
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    assert np.allclose(configuration.simbox.lengths, expected_box_vector)
    expected_number_of_particles = 32
    assert configuration['r'].shape[0] == expected_number_of_particles
    if verbose:
        print("positions:", configuration['r'])
        print("box_vector:", configuration.simbox.lengths)
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(configuration['r'][:, 0], configuration['r'][:, 1], configuration['r'][:, 2])
        plt.show()

def test_fcc_lattice_method(verbose=False, plot=False):
    conf = rp.Configuration()
    conf.make_lattice(rp.unit_cells.FCC, [2, 2, 2])
    positions = conf['r']
    box_vector = conf.simbox.lengths
    expected_positions = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [1.0, 1.0, 0.0], [1.5, 1.5, 0.0], [1.5, 1.0, 0.5], [1.0, 1.5, 0.5],
        [1.0, 0.0, 1.0], [1.5, 0.5, 1.0], [1.5, 0.0, 1.5], [1.0, 0.5, 1.5],
        [0.0, 1.0, 1.0], [0.5, 1.5, 1.0], [0.5, 1.0, 1.5], [0.0, 1.5, 1.5]])
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    #print(positions.size)
    #print(expected_positions.size)
    #assert np.allclose(positions, expected_positions, rtol=1e-4)
    #assert np.allclose(conf.simbox.lengths, expected_box_vector)
    print("positions:", positions)


def test_bcc_lattice(verbose=False, plot=False):
    cells = [2, 2, 2]
    positions, box_vector = rp.tools.make_lattice(rp.unit_cells.BCC, cells)
    configuration = rp.Configuration()
    configuration['r'] = positions
    configuration.simbox = rp.Simbox(configuration.D, box_vector)
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


def test_hexagonal(verbose=False, plot=False):
    cells = [4, 2]
    positions, box_vector = rp.tools.make_lattice(rp.unit_cells.HEXAGONAL, cells=cells)
    configuration = rp.Configuration()
    configuration['r'] = positions
    configuration.simbox = rp.Simbox(configuration.D, box_vector)
    expected_dimensions_of_space = 2
    assert configuration['r'].shape[1] == expected_dimensions_of_space
    expected_number_of_particles = 16
    assert configuration['r'].shape[0] == expected_number_of_particles

    if verbose:
        print('  Hexagonal lattice')
        print("positions:", configuration['r'])
        print("box_vector:", configuration.simbox.lengths)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Hexagonal lattice")
        plt.scatter(configuration['r'][:, 0], configuration['r'][:, 1])
        plt.axis('equal')
        plt.show()

def main():
    test_fcc_lattice(verbose=True, plot=True)
    test_fcc_lattice_method(verbose=True, plot=True)
    test_bcc_lattice(verbose=True, plot=True)
    test_hexagonal(verbose=True, plot=True)


if __name__ == "__main__":
    main()
