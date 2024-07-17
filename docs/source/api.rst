API
===

rumdpy (mainmodule)
-------------------

.. automodule:: rumdpy
   :members:
   :undoc-members:

The Simulation Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rumdpy.Simulation
   :members:
   :undoc-members:

The Configuration Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rumdpy.Configuration
   :members:

The Simbox Class
^^^^^^^^^^^^^^^^

.. autoclass:: rumdpy.Simbox
   :members:

The Evaluater Class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: rumdpy.Evaluater
   :members:

Integrators
-----------

.. autoclass:: rumdpy.NVE
   :members:

.. autoclass:: rumdpy.NVE_Toxvaerd
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

Interactions
-------------------

Pair potentials
^^^^^^^^^^^^^^^

.. automodule:: rumdpy.potential_functions
   :members:

Fixed interactions
^^^^^^^^^^^^^^^^^^

.. autofunction:: rumdpy.make_planar_calculator

.. autofunction:: rumdpy.setup_planar_interactions

Calculators
-----------

.. autoclass:: rumdpy.CalculatorRadialDistribution
   :members:

.. autoclass:: rumdpy.CalculatorStructureFactor
   :members:


Tools and helper functions
--------------------------

.. autofunction:: rumdpy.get_default_sim

.. autofunction:: rumdpy.generate_random_velocities

.. autofunction:: rumdpy.generate_fcc_positions

.. autofunction:: rumdpy.make_configuration_fcc

.. autofunction:: rumdpy.configuration_to_hdf5

.. autofunction:: rumdpy.hdf5_to_configuration

.. autofunction:: rumdpy.configuration_to_rumd3

.. autofunction:: rumdpy.configuration_from_rumd3

.. autofunction:: rumdpy.configuration_to_lammps

.. autofunction:: rumdpy.tools.make_lattice
