# Examples of using rumdpy

List of examples of atomic simulations in order of increasing complexity:
- minimal.py : Minimal example simulating a single component Lennard-Jones (LJ) system.
- blocks.py : Like minimal.py but specifying directly how simulation are performed in blocks.
- kablj.py : Simulating the Kob-Andersen binary LJ mixture. Also showing how to apply a temperature ramp for cooling.
- isochore.py : Performing several simulations in one script, here an isochore.
- isomorph.py : An isomorph is traced out using the gamma method. The script demomstrates the possibility of keeping the output of the simulation in memory (storage='memory'), usefull when a lot of short simulations are performed.
- 3Dviz.ipynb : Jupyter notebook demonstrating 3D visualization of cooling SC/KA Lennard-Jones using the package 'k3d'
