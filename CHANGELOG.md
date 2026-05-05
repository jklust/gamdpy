# Change log for `gamdpy`

## Version 0.8.3dev

### Bug fixes

* Fixed bug in make_lattice that gave wrong density for D!=3 and rho!=None.

### New features

* Integrator for Active Ornstein-Uhlenbeck Particle (gamdpy.ActiveOUP)
* Smooth Cubic spline truncations (gamdpy.apply_cubic_spline_cutoff)
* Smooth Gromacs style truncations (gamdpy.apply_gromacs_cutoff)
* Pair-potentials: Universal ZBL Potential, Gaussian-core model, Yukawa, Exponential repulsion, harmonic repulsion and Hertzian repulsion
* Tools for unit conversions (gp.conversion_factors)
* Tools for analyzing thermodynamics fluctuations

## Version 0.8.2, Aug 3, 2025

### Bug fixes

* Incorrect application of shifted-force cutoff solved

### New features

* Integrator for Brownian dynamics.
* Integrator for gradient descent.
* Integrator for NVU dynamics.
* Variable strain rate for SLLOD
* calc_dynamics can handle Lees-Edwards boundary conditions.
* Tabulated pair potentials.
* extract_scalars superseesed by ScalarSaver.extract(), see examples/read_scalar_data_from_h5.py
* TrajectorySaver enhanced to allow saving of velocities and forces.
* TimeScheduler's implemented to control when output is done (for now only in TrajectorySaver).
* examples/visualize.py for 3D visualization using ovito.
* examples/plot_pkls.py for plotting data from several simulations.

### Other

* Updates to output h5 format
* Update the format of the dictionary returned by CalculatorRadialDistribution.read()
  
## Version 0.8.1, Jun 12, 2025
First release of the package on pypi.
