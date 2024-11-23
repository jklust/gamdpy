#  **rumdpy [rum-dee-pai]** 

[The users guide to rumdpy](http://dirac.ruc.dk/~urp/2024/rumdpy/)

## Roskilde University Molecular Dynamics Python Package

Rumdpy implements molecular dynamics on GPU's in Python, relying heavily on the numba package ([numba.org](https://numba.pydata.org/)) which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda). 
The rumdpy package being pure Python (letting numba do the heavy lifting of generating fast code) results in an extremely extendable package: simply by interjecting Python functions in the right places, 
the (experienced) user can extend most aspect of the code, including: new integrators, new pair-potentials, new properties to be calculated during simulation, new particle properties, ...  

## NOTE
This is the developers version of the rumdpy package, NOT for general consumption just yet. Do NOT trust any of the results produced! Be prepared for interfaces and structure to change overnight. 

## Overall structure of the package

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

Temperature/Pressure can be controlled by a user-supplied function, see examples/kablj.py

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
 
## TODO, short term
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

## TODO, before summer interns arrive
- [X] SLLOD (stress, LEBC), Nick
- [X] Bonds interface
- [X] Implement other fixed interactions: point interactions (tethered particles). Jesper
- [ ] Finish Atomic interface (runtime actions...) Thomas
- [X] Momentum resetting (remove default) Nick
- [X] Read rumd3 & others configurations Nick
- [X] Testing (Framework, doctest), Ulf & Thomas
- [X] Testing using gitlab CI, Lorenzo
- [X] Include scalar column names in output, Lorenzo
- [X] Include vector column names in output, Lorenzo
- [X] Documentation/Tutorials/Best practices
- [ ] Reserve name on pypi, conda? Thomas
- [X] Generalize make_configuration to different lattices, Ulf
- [X] Read configurations from file (Lorenzo: added function load_output in tools)
- [ ] Runtime actions to include conf_saver and scalar_output, Thomas
- [X] Per particles thermostat using interaction
- [X] Post analysis, RDF and Sq 
- [X] Post analysis for multicomponents, Lorenzo/Danqui
- [ ] NVU integrator (tests missing), Mark

## Output Branch (branch origin/output)
### This branch is an attempt to make memory and disk output identical from user prospective
### Points to discuss/issues to address
- [X] sim.output is an h5py file, if in memory it uses driver='core'
- [X] sim.output can be generalized to be a different file type easily (only change in the Simulation().__init__)
- [ ] issue/feature 0: the output file is created when Simulation is initiated and stays there until close
- [ ] issue 1: the output file needs to be closed by the user before instantiating a new Simulation object
- [ ] fix   1: the Simulation object might create a new file every time (might cause memory problems)
- [ ] issue 2: there is an issue if two Simulation object are initialized at the same time with memory saving
- [ ] structure inside h5py: static info + a group for each evaluator

## TODO or decide not necessary, before paper/'going public'
- [X] Molecules (angles, dihedrals) Jesper
- [ ] Molecules (Interface) Jesper, Ulf
- [ ] Settle on io format
- [X] Implement O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [ ] Test O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [ ] Allow more flexible/dynamical changing which data to be stored in Configuration, Nick
- [ ] make GitLab/Hub address users, not ourselves (remove dev-state of page)
- [ ] make installable by pip for all, by uploading to pypi
- [ ] Use 'colarray' for scalars in Configuration (needs switching of dimensions)
- [ ] Configuration: include r_im in vectors
- [ ] Requirements/dependencies, especially to use grid-sync, ADD LINK NUMBA DOC 
- [ ] Auto-tuner, TBS
- [X] "grid to large for gridsync" should be handled ( CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE )
- [ ] Define hdf5 'template', discuss if h5md https://www.nongnu.org/h5md/ Lorenzo/output branch
- [ ] Ensure neighborlist integrity (automated check/reallocate)
- [ ] Benchmarking
- [ ] Charge (Water, SPCflexible), Jesper et al.


## TODO, long term:
- [ ] Constraints
- [ ] EAM metallic potentials
- [ ] Use sympy to differentiate pair-potentials. Was implemented but a factor of 2 slower, is float64's sneaking in?
- [ ] Add CPU support (can it be done as a decorator?)
- [ ] Thermostat on subsets of particles

## Various tools/strategies we will use
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- Git ( https://git-scm.com/doc, https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell ).
- Sphinx ( https://www.sphinx-doc.org/ ) for documentation, 
- ... to be hosted on readthedocs ( https://about.readthedocs.com/ ). Model: https://numba.readthedocs.io.
- Hypothesis (property based testing, https://hypothesis.readthedocs.io ).
- doctest (no more failing examples in examples/docs!, see colarray.py for example).
- Jupyter notebooks for tutorials. Testing: nbmake?, testbook?
- Automatic testing upon uploading (CI). How to get acces to GPU's?.
- Systematic benchmarking. Substantial degradation in performance will be considered a bug.

## Checklist for developing a new feature
- Copy code that resembles what you want to do, and modify it to your needs.
- Write tests in a file placed in tests (run pytest to check that it works).
- Write an example and place it in examples, add it to the examples/README.md
- Write documentation in the docstrings of the code (run doctests to check that it works).
- Include the new feature in the documentation, e.g. you may need to edit docs/source/api.rst

## Some git cmd which might be useful

Getting hash of your master (Head)
```sh
git log --pretty=format:'%h' -n 1
```

Creating a public branch (on the repository) starting from current master/branch
```sh
git checkout -b your_branch
git push -u origin your_branch
```

Difference in a single file between branches. Can use hash instead of master/branch
```sh
git diff origin branch -- rumdpy/Simulation.py
git diff hash1 hash2 -- rumdpy/Simulation.py
```
List the files that are different in two branches
```sh
git diff --name-only origin branch 
```
Show version of a file in another branch
```sh
git show branch:file
```

Reset last commit. It will not delete any file but will go back removing last commit and the add related to that commit
```sh
git reset HEAD~
```

## How to test the code
Running `pytest` in root (rumdpy) directory will run all tests.
This will use the settings in the file `pytest.ini`.

Install needed packages:

```sh
pip install pytest hypothesis scipy
```

Running pytest:

```sh
python3 -m pytest
```

Running all test typical takes several minutes.
Slow tests can be skipped by running (test functions decorated with `@pytest.mark.slow`):

```sh
python3 -m pytest -m "not slow"
```

Running pytest with -x option makes pytest stop after first failure
```sh
pytest -x
```

Running pytest starting from last failed test
```sh
pytest --lf
```

### Test of specific features

Test scripts are located in the `tests` directory. Most can be executed (in a verbose mode) as script:

```bash
python3 tests/test_examples.py
```

Running doctest of a single file:

```bash
python3 -m doctest -v rumdpy/calculators/calculator_radial_distribution.py
```

### Coverage of tests

To see what part of the code is covered:

```sh
pip install coverage
coverage run -m pytest
```

After the tests are finished do:

```sh
coverage report -m
```

or `coverage html`.

## Building documentation

To building the documentation using sphinx, https://www.sphinx-doc.org
(needs `pip install myst_nb pydata_sphinx_theme`)

Install needed packages:

```sh
pip install sphinx myst_nb pydata_sphinx_theme
```

Build documentation webpage:

```sh
cd docs
make html
```

Open webpage with firefox (or your favorite browers):

```sh
firefox build/html/index.html
```

Clean the build directory (optional):

```sh
make clean
```


# Known issues

## LinkerError: libcudadevrt.a not found
A workaround to fix the error `numba.cuda.cudadrv.driver.LinkerError: libcudadevrt.a not found` 
is to make a symbolic link to the missing file. 
This can be done by running the somthing like the below in the terminal:

```bash
ln -s /usr/lib/x86_64-linux-gnu/libcudadevrt.a .
```

in the folder of the script. Note that the path to `libcudadevrt.a` to the file may vary depending on the system.
