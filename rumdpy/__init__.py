#from .integrators import *
from .interactions import *
from .potential_functions import *
from .Configuration import *
from .Simulation import *
from .colarray import *
from .calculators import *
from .misc import *
from .visualization import *

# Trying to make eg. rp.integrators.nve available in user code, 
# without further imports than 'import rumdpy as rp'
# (similar to np.random.uniform)
# ... not working ...

import rumdpy.integrators as integrators
__all_ = list("integrators")
def __getattr__(attr):
    if attr == "integrators":
        import numpy.integrators as integrators
        return integrators