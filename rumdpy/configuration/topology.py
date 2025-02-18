import numpy as np
import numba
from numba import cuda

class Topology():
    """ 
    contains information about the topology, e.g. which bonds, angles and dihedrals are in the system
    """

    def __init__(self, molecule_names=None):
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.molecules = {}
        if molecule_names:
            for molecule_name in molecule_names:
                self.add_molecule_name(molecule_name)

    def add_molecule_name(self, name: str):
        self.molecules[name] = []

    # def read()
    # def write__to_hdf5()
    # ...

def bonds_from_positions(positions, cut_off, bond_type):
    bonds = []
    for i in range(len(positions)):
        for j in range(i):
            squared_distance = np.sum( (np.array(positions[j]) - np.array(positions[i]))**2 )
            if squared_distance <= cut_off**2:
                bonds.append([j, i, bond_type])
    return bonds

def angles_from_bonds(bonds, angle_type):
    angles = []
    for bond_index, bond in enumerate(bonds):
        for other_bond in bonds[bond_index+1:]:
            if bond[1] == other_bond[0]: # Assuming bonds inicies to be sorted!
                angles.append([bond[0], bond[1], other_bond[1], angle_type])
            elif other_bond[1] == bond[0]: # Assuming bonds inicies to be sorted!
                angles.append([other_bond[0], other_bond[1], other_bond[1], angle_type])
    return angles

def dihedrals_from_angles(angles, dihedral_type):
    dihedrals = []
    for angle_index, angle in enumerate(angles):
        for other_angle in angles[angle_index+1:]:
            if angle[1] == other_angle[0] and angle[2] == other_angle[1]: # Assuming bonds inicies to be sorted! WHAT IF CYCLIC?
                dihedrals.append([angle[0], angle[1], angle[2], other_angle[2], dihedral_type])
            elif other_angle[1] == angle[0] and other_angle[2] == angle[1]: # Assuming bonds inicies to be sorted!
                dihedrals.append([other_angle[0], other_angle[1], other_angle[2], angle[2], dihedral_type])
    return dihedrals

def molecules_from_bonds(bonds):
    molecules = []
    for bond in bonds:
        found = False
        for molecule in molecules:
            if bond[0] in molecule:
                found = True
                if bond[1] not in molecule:
                    molecule.append(bond[1])
                break
            if bond[1] in molecule:
                found = True
                if bond[0] not in molecule:
                    molecule.append(bond[0])
                break
        if not found:
            molecules.append(bond[0:2]) # If not bonded to existing molecule, the bond is part of a new molecule
    return molecules

def duplicate_topology(topology, num_molecules):
    new_topology = Topology()
    assert len(topology.molecules)==1 # Only one type (for now)
    for molecule_type in topology.molecules:
        assert len(topology.molecules[molecule_type])==1 # Only one molecule
        molecule_name = molecule_type
    new_topology.add_molecule_name(molecule_name)
    particles_per_molecule = len(topology.molecules[molecule_name][0])
    for molecule in range(num_molecules):
        first = molecule * particles_per_molecule
        for bond in topology.bonds:
            new_topology.bonds.append([bond[0] + first, bond[1] + first, bond[2]])
        for angle in topology.angles:
            new_topology.angles.append([angle[0] + first, angle[1] + first, angle[2]+ first, angle[3]])
        for dihedral in topology.dihedrals:
            new_topology.dihedrals.append(dihedral.copy()) # Copy needed?
            for index in range(4):
                new_topology.dihedrals[-1][index] += first
        new_topology.molecules[molecule_name].append([index + first for index in topology.molecules[molecule_name][0]]) 
    return new_topology


