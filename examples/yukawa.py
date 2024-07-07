""" Example of a user defined potential, example of a Yukawa potential.

This example uses a syntax similar to the backend of rumdpy, making it easy to
include the code in the package, and making it available to the community.

BUG: This example is not working, it is raising the following error:

  Use of unsupported NumPy function 'numpy.exp' or unsupported use of the function.

due to the use of numpy.exp in the yukawa function.

"""

import numpy as np
import matplotlib.pyplot as plt
import numba

import rumdpy as rp


def yukawa(dist, params):
    """ The Yukawa potential: u(r) = A·exp(-κ·r)/r

    parameters: κ, A    (κ is the greek letter kappa)

    The Yukawa potential is a simple screened Coulomb potential. The potential
    is used to model interactions between charged particles in a plasma. The
    potential is given by:

        u(r) = A·exp(-κ·r)/r

    where A is the strength of the interaction,
    and kappa is the inverse of the screening length.

    The s(r) function, used to compute pair forces (𝐅=s·𝐫), is defined as

        s(r) = -u'(r)/r

    and specifically for the Yukawa potential it is

        s(r) = A·exp(-κ·r)·(κ·r + 1)/r³

    The curvature (`d2u_dr2`) of the potential is given by

        u''(r) = A·exp(-κ·r)*([κ·r]² + 2κ·r + 2)/r³

    """
    # Extract parameters
    kappa = numba.float32(params[0])  # κ
    prefactor = numba.float32(params[1])  # A

    # Integers as floats
    one = numba.float32(1.0)
    two = numba.float32(2.0)
    three = numba.float32(3.0)

    # Compute helper variables
    kappa_dist = kappa * dist  # κ·r
    inv_dist = one / dist  # 1/r
    inv_dist3 = inv_dist ** three  # 1/r³
    exp_kappa_dist = prefactor * np.exp(-kappa_dist)  # A·exp(-κ·r)

    # Compute pair potential energy, pair force and pair curvature

    # A·exp(-κ·r)/r
    u = exp_kappa_dist * inv_dist

    # A·exp(-κ·r)·(κ·r + 1)/r³
    s = (kappa_dist + one) * exp_kappa_dist * inv_dist3

    # A·exp(-κ·r)*([κ·r]² + 2κ·r + 2)/r³
    d2u_dr2 = (kappa_dist ** two + two * kappa_dist + two) * exp_kappa_dist * inv_dist3

    return u, s, d2u_dr2  # u(r), s = -u'(r)/r, u''(r)


# Plot the Yukawa potential
plt.figure()
r = np.linspace(0.8, 3, 200, dtype=np.float32)
params = [1.0, 1.0, 2.5]
u, s, umm = yukawa(r, params)
plt.plot(r, u, label='u(r)')
plt.plot(r, s, label='s(r)')
plt.plot(r, umm, label='u\'\'(r)')
plt.xlabel('r')
plt.ylabel('u, s, u\'\'')
plt.legend()
plt.show()

# Setup configuration: FCC Lattice
configuration = rp.Configuration()
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(T=0.7)

# Setup pair potential: Single component Yukawa system
pair_func = rp.apply_shifted_potential_cutoff(yukawa)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = rp.PairPotential2(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVE(dt=0.005)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=32,
                    steps_per_timeblock=1024,
                    storage='memory')

# Run simulation
sim.run()
