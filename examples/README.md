# Examples

List of examples of simulations (mostly in order of increasing complexity).

## Basic

- [minimal.py](minimal.py) : Minimal example simulating a single component Lennard-Jones (LJ) system. 
- [blocks.py](blocks.py) : Like minimal.py but specifying directly how simulation are performed in blocks.

## Analysis

- [thermodynamics.py](thermodynamics_old.py) : Calculate thermodynamic properties.
- [isochore.py](isochore.py) : Performing several simulations in one script, here an isochore.
- [isomorph.py](isomorph.py) : An isomorph is traced out using the gamma method. The script demonstrates the possibility of keeping the output of the simulation in memory (storage='memory'), usefull when a lot of short simulations are performed.
- [test_stress.py](test_stress.py) : Like blocks.py but calculates also the stress tensor and prints it after each block.
- [structure_factor.py](structure_factor.py) : Calculate the structure factor of a Lennard-Jones system and plot it.
- [3Dviz.ipynb](3Dviz.ipynb) : Jupyter notebook demonstrating 3D visualization of cooling SC/KA Lennard-Jones using the package 'k3d'

## Atomistic models

- [bcc_lattice.py](bcc_lattice.py) : How to set up other initial structures (bcc lattice of LJ particles).
- [2D.py](2D.py) : Simulating a 2D system (Lennard-Jones particle in a hexagonal lattice).
- [kablj.py](kablj.py) : Simulating the Kob-Andersen binary LJ mixture. Also showing how to apply a temperature ramp for cooling.
- [tethered_particles.py](tethered_particles.py) : Simulation of tethered particles.
- [atomistic_walls.py](atomistic_walls.py) : Simulation of an atomistic wall, defined by tethered and thermostated particles.
- [yukawa.py](yukawa.py) : Example of implementing a user-defined potential, exemplified by the Yukawa potential.

## Molecular models

- [ASD.py](ASD.py) : ASymmetric Dumbbels (toy model of toluene)
- [LJchain.py](LJchain.py) : LJ chains (coarse grained polymer model)

## Input and output

Input and output can be done in formats that are not directly supported by the package.

- [write_to_lammps.py](write_to_lammps.py) : Write dump file in LAMMPS format. Can be open by external tools like OVITO or VMD.
