Features
========

Please note that this is an early stage of development of the rumdpy package,
and that it is not for general consumption just yet. 
Do not trust the results produced, 
and be prepared for interfaces and structure to change overnight.

Integrators
-----------

A function (or kernel) that takes a configuration as input and takes a number of time steps. Interactions to be used are compiled in when constructed (make_integrator()).
- NVE,
- NVT (Langevin and Nose-Hoover)
- NPT (Langevin)

Interactions
------------

A function (or kernel) that takes a configuration as input and computes forces and other properties as requested during its construction (make_interactions()). 
The interaction function/kernel is responsible for keeping any internal datastructures up to date (in particular: nblist). 
- pairpotential (parameters, nblist, ...)
- fixed interactions (interactions known beforehand): 
  - bonds (angles, dihedrals to be implemented)
  - planar interactions: smooth walls, gravity, electric fields, ...
  - point interactions, e.g. tethering (to be implemented)

Evaluator
---------

Takes a configuration and an interactions function/kernel, and evaluates properties as specified at its construction (make_evaluator)
