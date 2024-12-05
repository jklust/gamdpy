# Examples

List of examples of simulations (mostly in order of increasing complexity).

## Basic

- [minimal.py](minimal.py) : Minimal example simulating a single component Lennard-Jones (LJ) system using the NVT integrator. 
- [minimal_cpu.py](minimal_cpu.py) : Like minimal.py but running on the cpu using numba CUDA simulator.
- [minimal_NPT.py](minimal_NPT.py) : Minimal example simulating a single component Lennard-Jones (LJ) system using the NPT integrators.
- [blocks.py](blocks.py) : Like minimal.py but specifying directly how a simulation is performed in blocks.

## Analysis

- [thermodynamics.py](thermodynamics.py) : Calculate thermodynamic properties.
- [isochore.py](isochore.py) : Performing several simulations in one script, here an isochore.
- [isomorph.py](isomorph.py) : An isomorph is traced out using the gamma method. The script demonstrates the possibility of keeping the output of the simulation in memory (storage='memory'), useful when a lot of short simulations are performed.
- [test_stress.py](test_stress.py) : Like blocks.py but calculates also the stress tensor and prints it after each block.
- [structure_factor.py](structure_factor.py) : Calculate the structure factor of a Lennard-Jones system and plot it.
- [3Dviz.ipynb](3Dviz.ipynb) : Jupyter notebook demonstrating 3D visualization of cooling SC/KA Lennard-Jones using the package 'k3d'.
- [consistency_NPT.py](consistency_NPT.py) Calcuate several thermodynamic quantities (dP/dT|<sub>V</sub>, $\beta$<sub>P</sub>, c<sub>V</sub>, c<sub>P</sub>, K<sub>T</sub>) and check consistency of NVT and NPT fluctuations. 
- [widoms_particle_insertion.py](widoms_particle_insertion.py) Widom's particle insertion method for calculating the chemical potential.
- [evaluator_inverse_powerlaw.py](evaluator_inverse_powerlaw.py) Simulate Lennard-Jones system and evaluate the inverse power law potential.
- [evaluator_einstein_crystal.py](evaluator_einstein_crystal.py) Simulate Lennard-Jones crystal and evaluate the harmonic tether potential.

## Atomistic models

- [bcc_lattice.py](bcc_lattice.py) : How to set up other initial structures (bcc lattice of LJ particles).
- [2D.py](2D.py) : Simulating a 2D system (Lennard-Jones particle in a hexagonal lattice).
- [kablj.py](kablj.py) : Simulating the Kob-Andersen binary LJ mixture. Also showing how to apply a temperature ramp for cooling.
- [tethered_particles.py](tethered_particles.py) : Simulation of tethered particles.
- [poiseuille.py](poiseuille.py) : Simulation of a nano-scale Poiseuille flow in a slit-pore.
- [yukawa.py](yukawa.py) : Example of implementing a user-defined potential, exemplified by the Yukawa potential.
- [hydrocorr.py](hydrocorr.py) : Calculation of the hydrodynamic correlation function for a LJ liquid

## Molecular models

- [ASD.py](ASD.py) : ASymmetric Dumbbells (toy model of toluene).
- [LJchain.py](LJchain.py) : LJ chains (coarse grained polymer model).
- [generic_molecules.py](generic_molecules.py): Simulation of a linear molecule with bond, angle, and dihedral potentials

## Input and output

Input and output can be done in formats that are not directly supported by the package.

- [write_to_lammps.py](write_to_lammps.py) : Write dump file in LAMMPS format. Can be open by external tools like OVITO or VMD.

## Post-processing

Examples of postprocessing existing rumdpy/rumd3 data. This scripts require the output produced with minimal.py or a rumd3 TrajectoryFile.

- [calc_rdf_from_h5.py](calc_rdf_from_h5.py) : Read a simulation saved as h5 and calculate the RDF.
- [calc_rdf_from_rumd3.py](calc_rdf_from_rumd3.py) : Read a simulation saved as "TrajectoryFiles" from rumd3 and calculate the RDF.
- [calc_sq_from_h5.py](calc_sq_from_h5.py) : Read a simulation saved as h5 and calculate the structure factor S(q).

## Other
- [NVU_RT_kob_andersen.py](NVU_RT_kob_andersen.py) : Use NVU ray tracing integrator
