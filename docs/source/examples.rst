Examples
========

Minimal
-------

Minimal example simulating a single component Lennard-Jones (LJ) system.

.. literalinclude:: ../../examples/minimal.py
    :language: python

Features
--------

Blocks
^^^^^^

Like minimal.py but specifying directly how simulation are performed in blocks

.. literalinclude:: ../../examples/blocks.py
    :language: python

Isochore
^^^^^^^^

Performing several simulations in one script, here an isochore.

.. literalinclude:: ../../examples/isochore.py
    :language: python

Isomorph
^^^^^^^^

An isomorph is traced out using the gamma method. The script demomstrates the possibility of keeping the output of the simulation in memory (storage='memory'), usefull when a lot of short simulations are performed.

.. literalinclude:: ../../examples/isomorph.py
    :language: python


Structure factor
^^^^^^^^^^^^^^^^

Calculate the structure factor of a Lennard-Jones system and plot it.

.. literalinclude:: ../../examples/structure_factor.py
    :language: python

The BCC lattice
^^^^^^^^^^^^^^^

Simulating the BCC lattice of a Lennard-Jones system.

.. literalinclude:: ../../examples/bcc_lattice.py
    :language: python


Models
------


Binary mixture
^^^^^^^^^^^^^^

Simulating the Kob-Andersen binary LJ mixture. Also showing how to apply a temperature ramp for cooling.

.. literalinclude:: ../../examples/kablj.py
    :language: python

