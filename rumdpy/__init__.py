""" The rumdpy main module """
# Objects which are imported here will be in the main namespace and can be called are rumdpy.object
# Objects which are imported in the __init__.py of subpackages are called as rumdpy.subpackage.object

# Import from configuration subpackage
# The configuration subpackage contains details about the configuration (positions, energies, etc)
# The class Simbox has informations about the simulation box
from .configuration.Configuration import *
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
from .integrators import *

# Import from interactions subpackage
from .interactions import *
from .interactions.potential_functions import *

# Import from calculators subpackage
# Calculators are runtime actions with interact with the kernel
from .calculators import *

# Import from tools subpackage
# Tools/Evaluator are runtime actions with do not interact with the kernel
from .tools.Evaluator import Evaluator
# load_output, save_configuration and calc_dynamics are imported in configuration/__init__.py

# Import from misc
# Misc folder contains scripts that have no better place in the code
from .misc.plot_scalars import plot_scalars
from .misc.make_function import make_function_constant, make_function_ramp, make_function_sin

# Import from visualization 
#from .visualization import *


__version__ = "0.0.1"
