import numpy as np
import numba
from numba import cuda

class Topology():
    """ 
    contains information about the topology, e.g. which bonds, angles and dihedrals are in the system
    """

    def __init__(self):
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.molecules = {}

    def add_molecule_name(self, name: str):
        self.molecules[name] = []

    # def read()
    # def write__to_hdf5()
    # ...

