#  **rumdpy [rum-dee-pai]** 
## Roskilde University Molecular Dynamics Python Package

## NOTE:
This is the developers version of the rumdpy package, not for generaæ consumption just yet. Rumdpy implements molecular dynamics on GPU's in Python, relying heavely on the numba package which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda). 

## Overall structure of the package:

### 1. Configuration
A class containing all relevant information about a configuration. Methods are kept to a minimum, relying instead on stand-alone functions taking configurations as input.
- Vectors (r, v, f, etc) N,D float array storing D-dimensional vector for each particle 
- Scalars (mass, kinetic energy, etc)  N, float array storing scalar each particle 
- sim_box (data + functions)

### 2. Integrator
A function (or kernel) that takes a configuration as input and takes a number of timesteps. Interactions to be used are compiled in when contructed (make_integrator()).
- NVE, 
- NVT, 
- ...

### 2. Interactions
A function (or kernel) that takes a configuration as input and computes forces and other properties as requested during its construction (make_interactions()). The interaction function/kernel is responsible for keeping any internal datastructures up to date (in particular: nblist). 
- pairpotential (parameters, nblist, ...)
- bonds etc (not implemented yet)

### 4. Evaluator
Takes a configuration and an interactions function/kernel, and evaluates properties as specified at its construction (make_evaluator)
- not implemented yet

## Implementing on GPU using numba.cuda

- Inherited from rumd3: pb (particles per block), tp (threads per particle)
- Hoping to avoid from rumd3: sorting (gets too complicated).

Syncronization is of outmost importance for correctness. For example, all forces needs to be calculated before the integrator starts moving the particles. Traditionally (and in rumd3) this is done by kernel-calls: it is guarenteed that one kernel finsihes before the next begins (unless you explicitly ask otherwise). 

Unfortunately kernel calls are slow, especially in numba.cuda (as compared to c++.cuda). A rough estimate is that the maximum number of timesteps per second (TPS) that can be achieved using kernel calls for syncronization is about 5000 - a far cry from the ~100.000 TPS that can be achieved for small systems using "grid syncronization": Calling 'grid.sync()' inside a kernel ensures all threads in the grid gets syncronised (i.e., no threads proceeds beyond this statement before all threads have reached this statement). 

There is a limit to how many thread blocks can be used with grid syncronization, which makes it inefficient at large system sizes, so we need to be able to chose between the two ways of synconization. A good place to see how this is done without implementing all functions twice is in 'integrators.py'
 

## TODO, short term
- [x] Break single file into several files/modules 
- [ ] Start using GIT
- [ ] Make it into a python package that can be installed locally by pip
- [ ] Implement (FENE) spring, and settle on structure for defining interactions (beyond pair potentials)
- [ ] Implement exlusion list
- [ ] Implement O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [ ] upload to GitLab/Hub
- [ ] Use sympy to differential pair-potentials. Was implemented but a factor of 2 slower, is float64's sneaking in?
- [ ] Autotuner
- [ ] Add CPU support (can it be done as a decorator?)

## TODO, long term
- [ ] make public GitLab/Hub page
- [ ] upload to pypi
