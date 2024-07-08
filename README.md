#  **rumdpy [rum-dee-pai]** 
## Roskilde University Molecular Dynamics Python Package

Rumdpy implements molecular dynamics on GPU's in Python, relying heavily on the numba package ([numba.org](https://numba.pydata.org/)) which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda). 
The rumdpy package being pure Python (letting numba do the heavy lifting of generating fast code) results in an extremely extendable package: simply by interjecting Python functions in the right places, 
the (experienced) user can extend most aspect of the code, including: new integrators, new pair-potentials, new properties to be calculated during simulation, new particle properties, ...  

## NOTE:
This is the developers version of the rumdpy package, NOT for general consumption just yet. Do NOT trust any of the results produced! Be prepared for interfaces and structure to change overnight. 

## Overall structure of the package:

### 1. Configuration
A class containing all relevant information about a configuration, including the simulation box (class sim_box). 
- Vectors (r, v, f, etc): (N,D) float array storing D-dimensional vector for each particle 
- Scalars (mass, kinetic energy, etc) (N,) float array storing scalar each particle 
- sim_box (data describing box + functions implementing how to calculate distances and how to implement BC). For now: only Cuboid box.  

### 2. Integrators
Classes implementing a simulation algorithm. Currently implemented: 
- class NVE 
- class NVE_Toxvaerd
- class NVT : Nose-Hoover thermostat 
- class NVT_Langevin
- class NPT_Langevin
... all integrators converted to classes to use new interface. Old ones kept briefly for testing.

Temperature can be controlled by a user-supplied function, see examples/kablj.py

### 3. Interactions
Classes implementing interactions that can be applied to particles in the system:  
- class PairPotential (stores potential parameters and the neighbour list to use (class NbList)
- fixed interactions (interactions known beforehand): 
  - bonds (angles, dihedrals to be implemented)
  - planar interactions: smooth walls, gravity, electric fields, ...
  - point interactions, e.g. tethering (to be implemented)

An interaction is responsible for keeping any internal datastructures up to date (in particular: class PairPotential is resposible for keeping its neighbor-list (class NbList up to date). 

### 4. class Simulation
Takes a Configuration, an Integrator, and a (list of) Interaction(s) and sets up a simulation. Also controls when momentum-resetting and output is performed. Performing simulation is done by a method of this class.

### 5. Evaluator
Takes a Configuration and a (list of) Interaction(s), and evaluates properties.

# Info for developers

## Implementing on GPU using numba.cuda

- Inherited from rumd3: pb (particles per block), tp (threads per particle)
- Hoping to avoid from rumd3: sorting (gets too complicated).
- To be implemented: Autotuner. For now we are relying on default parameters dependent on number of particles and number of core on GPU (see misc.py/get_default_compute_plan).

Synchronization is of utmost importance for correctness. For example, all forces needs to be calculated before the integrator starts moving the particles. 
Traditionally (and in rumd3) this is done by kernel-calls: it is guaranteed that one kernel finishes before the next begins (unless you explicitly ask otherwise). 

Unfortunately kernel calls are slow, especially in numba.cuda (as compared to c++.cuda). 
A rough estimate is that the maximum number of time steps per second (TPS) that can be achieved using kernel calls for synchronization is about 5000 - a far cry from the ~100.000 TPS that can be achieved for small systems using "grid synchronization": Calling 'grid.sync()' inside a kernel ensures all threads in the grid gets syncronised (i.e., no threads proceeds beyond this statement before all threads have reached this statement). 

There is a limit to how many thread blocks can be used with grid synchronization, which makes it inefficient at large system sizes, so we need to be able to chose between the two ways of synchronization. 
A good place to see how this is done without implementing all functions twice is in 'integrators.py'
 
 ## TODO, short term:
- [x] Break single file into several files/modules 
- [x] Start using GIT
- [x] Make it into a python package that can be installed locally by pip
- [x] cut = 2.5 hardcoded - change that! -> 'max_cut' now part of interaction parameters for pair-potential 
- [x] Implement springs, as an example of 'fixed interactions' (needs testing for gridsync==False). 
- [x] Implement (fixed) planar interactions, eg. smooth walls, gravity, and electric fields.
- [x] Implement exclusion list 
- [x] upload to GitLab
- [x] Use 'colarray' for vectors in Configuration
- [x] Move r_ref from Configuration to nblist

## TODO, before summer interns arrive:
- [ ] SLLOD (stress, LEBC), Nick
- [X] Bonds interface
- [X] Implement other fixed interactions: point interactions (tethered particles). Jesper
- [ ] Finish Atomic interface (runtime actions...) Thomas
- [X] Momentum resetting (remove default) Nick
- [X] Read rumd3 & others configurations Nick
- [X] Testing (Framework, doctest), Ulf & Thomas
- [ ] Testing using gitlab CI
- [ ] Settle on io format
- [ ] Documentation/Tutorials/Best practices
- [ ] Reserve name on pypi, conda?
- [X] Generalize make_configuration to different lattices, Ulf
- [ ] Read configurations from file
- [ ] Allow more flexible/dynamical changing which data to be stored in Configuration (e.g. stresses)
- [ ] Runtime actions to include conf_saver and scalar_output, Thomas
- [ ] NVT_Langevin-per-particle integrator, Jesper
- [ ] Post analysis, RDF and Sq
- [ ] NVU integrator, Mark

## TODO or decide not necesarry, before paper:
- [ ] Molecules (angles, dihedrals, Interface) Jesper, Ulf
- [ ] Implement O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [ ] make GitLab/Hub address users, not ourselves (remove dev-state of page)
- [ ] make installable by pip for all, by uploading to pypi
- [ ] Use 'colarray' for scalars in Configuration (needs switching of dimensions)
- [ ] Configuration: include r_im in vectors
- [ ] Requirements/dependencies, especially to use grid-sync 
- [ ] Auto-tuner
- [ ] "grid to large for gridsync" should be handled ( CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE )
- [ ] Define hdf5 'template' (discuss if h5md https://www.nongnu.org/h5md/)
- [ ] Ensure neighborlist integrity (automated check/reallocate)

## TODO, long term:
- [ ] Constraints
- [ ] Use sympy to differentiate pair-potentials. Was implemented but a factor of 2 slower, is float64's sneaking in?
- [ ] Add CPU support (can it be done as a decorator?)

## Various tools/strategies we will use:
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- Git ( https://git-scm.com/doc, https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell ).
- Sphinx ( https://www.sphinx-doc.org/ ) for documentation, 
- ... to be hosted on readthedocs ( https://about.readthedocs.com/ ). Model: https://numba.readthedocs.io.
- Hypothesis (property based testing, https://hypothesis.readthedocs.io ).
- doctest (no more failing examples in examples/docs!, see colaaray.py for example).
- Jupyter notebooks for tutorials. Testing: nbmake?, testbook?
- Automatic testing upon uploading (CI). How to get acces to GPU's?.
- Systematic benchmarking. Substantial degradation in performance will be considered a bug.

## Notes on how to test the code
Run `pytest` in root (rumdpy) directory.
NOTE: pytest fails if k3d not installed

## Known issues:

### LinkerError: libcudadevrt.a not found
A workaround to fix the error `numba.cuda.cudadrv.driver.LinkerError: libcudadevrt.a not found` 
is to make a symbolic link to the missing file. 
This can be done by running the somthing like the below in the terminal:
```bash
ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .
```
in the folder of the script. Note that the path to `libcudadevrt.a` to the file may vary depending on the system.

# Installation (in progress)

## Installing rumdpy on windows using Windows Subsystem For Linux (WSL)

### Install WSL
Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator", enter the command
- wsl --install     
press enter and then restart your machine. 
The default installation is Ubuntu, for others check: https://learn.microsoft.com/en-us/windows/wsl/install

### Install python and pip on WSL

- open Windows Command Prompt
- in the tab bar click on "v" and select ubuntu 
- sudo apt-get update
- sudo apt-get install python3.10
- sudo apt-get install pip

### Install miniconda https://docs.anaconda.com/miniconda/

- mkdir -p ~/miniconda3
- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
- bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
- rm -rf ~/miniconda3/miniconda.sh
- ~/miniconda3/bin/conda init bash
 
### Install cuda

- miniconda3/condabin/conda install cudatoolkit
- sudo apt install nvidia-cuda-toolkit
- modify .bashrc adding: export LD_LIBRARY_PATH="/usr/lib/wsl/lib/" from https://github.com/numba/numba/issues/7104

### Install rumdpy

- pip install git+https://gitlab.com/tbs.cph/rumdpy-dev.git

## Installing rumdpy on windows using Anaconda

WARNING: due to naming of the integrators the package will not work without renaming/editing some files (awaiting fix)

### Install Anaconda
### Install rumdpy (and pip) using Powershell Prompt in Anaconda:
- open Anaconda Powershell as admin (from search) 
- conda update -n base -c defaults conda
- conda install anaconda::pip
- conda install anaconda::git
- conda config --set channel_priority flexible
- conda install cudatoolkit
- pip install git+https://gitlab.com/tbs.cph/rumdpy-dev.git

