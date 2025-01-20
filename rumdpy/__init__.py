""" The rumdpy main module """
# Objects which are imported here will be in the main namespace and can be called are rumdpy.object
# Objects which are imported in the __init__.py of subpackages are called as rumdpy.subpackage.object

# Import from configuration subpackage
# The configuration subpackage contains details about the configuration (positions, energies, etc)
# The class Simbox has informations about the simulation box
from .configuration.Configuration import Configuration
from .configuration.Configuration import configuration_to_hdf5, configuration_from_hdf5, configuration_to_rumd3, configuration_from_rumd3, configuration_to_lammps
from .configuration.Simbox import Simbox, Simbox_LeesEdwards
from .configuration.colarray import colarray 
from .configuration import unit_cells
# make_lattice is imported in configuration/__init__.py

# Import from simulation subpackage
from .simulation.Simulation import Simulation
from .simulation.get_default_sim import get_default_sim
from .simulation.get_default_compute_plan import get_default_compute_plan
from .simulation.get_default_compute_flags import get_default_compute_flags
# Import from integrators subpackage
from .integrators import integrator, NVE, NVT, NVT_Langevin, NPT_Atomic, NPT_Langevin, SLLOD, NVU_RT

# Import from interactions subpackage
from .interactions import *
from .interactions.potential_functions import *
#from .interactions import interaction, nblist, nblist_linked_lists
#from .interactions import PairPotential#pair_potential
#from .interactions import bonds, angles, dihedrals
#from .interactions import make_fixed_interactions, gravity, planar_interactions, relaxtemp, tether

# Import from calculators subpackage
# Calculators are runtime actions with interact with the kernel
from .calculators import RuntimeAction, add_runtime_actions_list, ConfSaver, ScalarSaver, MomentumReset
from .calculators import CalculatorHydrodynamicCorrelations, CalculatorHydrodynamicProfile, CalculatorWidomInsertion
from .calculators import CalculatorRadialDistribution, CalculatorStructureFactor

# Import from tools subpackage
# To make type checking work (e.g. pylance): 
from .tools import TrajectoryIO, calc_dynamics, save_configuration
# Side effect rp.calc_dynamics does also work! Same problem for integrators
# TrajectoryIO save_configuration and calc_dynamics are not directly imported and are called via rp.tools.*

# Tools/Evaluator are runtime actions with do not interact with the kernel
from .tools.Evaluator import Evaluator

# Import from misc
# Misc folder contains scripts that have no better place in the code
from .misc.plot_scalars import plot_scalars
from .misc.make_function import make_function_constant, make_function_ramp, make_function_sin
from .misc.extract_scalars import extract_scalars

# Import from visualization 
#from .visualization import *

__version__ = "0.0.1"
