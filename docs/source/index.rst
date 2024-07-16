.. rumdpy documentation master file, created by
   sphinx-quickstart on Thu Feb  8 13:45:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About rumdpy
============

Roskilde University Molecular Dynamics Python Package (rumdpy) implements molecular dynamics on GPU's in Python, relying heavily on the numba package (https://numba.pydata.org/) which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda).
The rumdpy package being pure Python (letting numba do the heavy lifting of generating fast code) results in an extremely extendable package: simply by interjecting Python functions in the right places,
the (experienced) user can extend most aspect of the code, including: new integrators, new pair-potentials, new properties to be calculated during simulation, new particle properties, etc.

Here is an example of a script that runs a simple Lennard-Jones simulation ( :download:`minimal.py <./examples/minimal.py>`):

.. literalinclude:: ./examples/minimal.py
   :language: python

Features
--------

* Pure Python code (with numba JIT compilation)

  * Easy to extend
  * Fast

* Numerical integrators

  * Leapfrog NVE, Nose-Hoover NVT, Langevin NVT, Langevin NpT

* Atom systems

  * Lennard-Jones
  * Multi component systems
  * Easy to extend to any pair-potential
  * Scalar forces such as gravity or tethered particles.

* Molecular systems

  * Harmonic bonds
  * Easy to extend to any bond-potential

* Analysis tools

  * Structure: radial distribution function, structure factor
  * Dynamics: mean square displacement

* Easy integration with other codebases

  * Easy analysis with popular Python packages such as numpy, pandas, matplotlib, etc.
  * Read and write RUMD3 files and write LAMMPS data files.


Contents
========

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   examples/README.md
   api
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

