# Examples of using rumdpy

List of examples of atomic simulations in order of increasing complexity:
- [minimal.py](minimal.py) : Minimal example simulating a single component Lennard-Jones (LJ) system. 
- [blocks.py](blocks.py) : Like minimal.py but specifying directly how simulation are performed in blocks.
- [kablj.py](kablj.py) : Simulating the Kob-Andersen binary LJ mixture. Also showing how to apply a temperature ramp for cooling.
- [isochore.py](isochore.py) : Performing several simulations in one script, here an isochore.
- [isomorph.py](isomorph.py) : An isomorph is traced out using the gamma method. The script demomstrates the possibility of keeping the output of the simulation in memory (storage='memory'), usefull when a lot of short simulations are performed.
- [test_stress.py](test_stress.py) : Like blocks.py but calculates also the stress tensor and prints it after each block.
- [3Dviz.ipynb](3Dviz.ipynb) : Jupyter notebook demonstrating 3D visualization of cooling SC/KA Lennard-Jones using the package 'k3d'
- [structure_factor.py](structure_factor.py) : Calculate the structure factor of a Lennard-Jones system and plot it.
- [ASD.py](ASD.py) : Simulating ASymmetric Dumbbels (toy model of toluene)
- [LJchain.py](LJchain.py) : Simulating LJ chains (coarse grained polymer model)
