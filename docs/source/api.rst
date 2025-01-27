API
###


The Simulation Class
********************

.. autoclass:: rumdpy.Simulation
   :members:
   :undoc-members:

The Configuration Class
***********************

.. autoclass:: rumdpy.Configuration
   :members:
   :undoc-members:

Integrators
***********

.. autoclass:: rumdpy.NVE
   :members:

.. autoclass:: rumdpy.NVT
   :members:

.. autoclass:: rumdpy.NVT_Langevin
   :members:

.. autoclass:: rumdpy.NPT_Atomic
   :members:

.. autoclass:: rumdpy.NPT_Langevin
   :members:

.. autoclass:: rumdpy.SLLOD
   :members:

.. autoclass:: rumdpy.NVU_RT
   :members:

Interactions
************

Pair potentials
===============

.. autoclass:: rumdpy.PairPotential
   :members:

Functions (pair potentials)
---------------------------

.. autofunction:: rumdpy.LJ_12_6

.. autofunction:: rumdpy.LJ_12_6_sigma_epsilon

.. autofunction:: rumdpy.harmonic_repulsion

.. autofunction:: rumdpy.hertzian

.. autofunction:: rumdpy.SAAP

Functions (bonds)
---------------------------

.. autofunction:: rumdpy.harmonic_bond_function

Generators
----------

Generators return a function that can be used to calculate the potential energy and the force between two particles.

.. autofunction:: rumdpy.make_LJ_m_n

.. autofunction:: rumdpy.make_IPL_n

.. autofunction:: rumdpy.make_potential_function_from_sympy

Modifies
--------

Modifies are typically used to smoothly truncate the potential at a certain distance.

.. autofunction:: rumdpy.apply_shifted_potential_cutoff

.. autofunction:: rumdpy.apply_shifted_force_cutoff

Fixed interactions
==================

Classes
-------

.. autoclass:: rumdpy.Bonds

.. autoclass:: rumdpy.Angles

.. autoclass:: rumdpy.Tether

.. autoclass:: rumdpy.Gravity

.. autoclass:: rumdpy.Relaxtemp

Generators
----------

.. autofunction:: rumdpy.make_planar_calculator

.. autofunction:: rumdpy.setup_planar_interactions

.. autofunction:: rumdpy.make_fixed_interactions


Calculators
***********

.. autoclass:: rumdpy.CalculatorRadialDistribution
   :members:

.. autoclass:: rumdpy.CalculatorStructureFactor
   :members:

.. autoclass:: rumdpy.CalculatorWidomInsertion
   :members:

.. autoclass:: rumdpy.CalculatorHydrodynamicCorrelations
   :members:

.. autoclass:: rumdpy.CalculatorHydrodynamicProfile
   :members:

Tools and helper functions
**************************

Input and Output
================

The TrajectoryIO class
----------------------

.. autoclass:: rumdpy.tools.TrajectoryIO
   :members:
   :undoc-members:

IO functions
------------

.. autofunction:: rumdpy.tools.save_configuration

.. autofunction:: rumdpy.configuration_to_hdf5

.. autofunction:: rumdpy.configuration_from_hdf5

.. autofunction:: rumdpy.configuration_to_rumd3

.. autofunction:: rumdpy.configuration_from_rumd3

.. autofunction:: rumdpy.configuration_to_lammps


Mathematical functions
======================

The below returns functions that can be executed fast in a GPU kernel.
As an example, they can be used to set a time-dependent target temperature.

.. autofunction:: rumdpy.make_function_constant

.. autofunction:: rumdpy.make_function_ramp

.. autofunction:: rumdpy.make_function_sin

Extract data
============

.. autofunction:: rumdpy.extract_scalars

Miscellaneous
*************

.. autofunction:: rumdpy.get_default_sim

.. autofunction:: rumdpy.get_default_compute_plan

.. autofunction:: rumdpy.get_default_compute_flags
