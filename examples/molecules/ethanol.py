# Load parameters for a simulation of ethanol
#
# TraPPE force-field parameters for ethanol is downloaded at, http://trappe.oit.umn.edu/
#
# The pdf file is from https://github.com/wesbarnett/OPLS-molecules/tree/master/pdb/alcohols
#
# wget https://raw.githubusercontent.com/wesbarnett/OPLS-molecules/master/pdb/alcohols/ethanol.pdb

import xml.etree.ElementTree as ET
from pprint import pprint


def load_pdb(filename: str) -> dict:
    """ Load a PDB file and return a lists of Atom name, Residue name and coordinates"""
    atom_name = []
    residue_name = []
    coordinates = []
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name.append(line[12:16].strip())
                residue_name.append(line[17:20].strip())
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {
        "atom_name": atom_name,
        "residue_name": residue_name,
        "coordinates": coordinates
    }


def strip_hydrogen_on_carbons(molecule: dict) -> dict:
    """ Return a molecule dictionary without hydrogen atoms on carbons
    The dictionary should have atom_name, residue_name and coordinates.

    A carbon is identified as C* and a hydrogen is identified as H*.
    A connecting hydrogen should be below the carbon atom before next non-hydrogen atom.
    The resulting dictionary have a count of hydrogens attaced to a carbon atom """

    # Count hydrogens attached to a carbon atom. Break if atom is non-hydrogen
    carbon_hydrogens = {}
    hydrogen_to_be_stripped = [False] * len(molecule["atom_name"])
    for i, atom in enumerate(molecule["atom_name"]):
        if atom.startswith("C"):
            carbon_hydrogens.update({molecule["atom_name"][i]: 0})
            for j in range(i + 1, len(molecule["atom_name"])):
                if molecule["atom_name"][j].startswith("H") and molecule["residue_name"][j] == molecule["residue_name"][
                    i]:
                    carbon_hydrogens[molecule["atom_name"][i]] += 1
                    hydrogen_to_be_stripped[j] = True
                else:
                    break

    # Remove hydrogens attached to a carbon atom, but not to other atoms
    atom_name = []
    residue_name = []
    coordinates = []
    for i, atom in enumerate(molecule["atom_name"]):
        if not hydrogen_to_be_stripped[i]:
            atom_name.append(atom)
            residue_name.append(molecule["residue_name"][i])
            coordinates.append(molecule["coordinates"][i])

    return {
        "atom_name": atom_name,
        "residue_name": residue_name,
        "coordinates": coordinates,
        "carbon_hydrogens": carbon_hydrogens
    }


def load_parameters(filename: str) -> dict:
    # Load and parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Initialize a dictionary to store XML data
    data = {
        "model": root.attrib["model"],
        "AtomTypes": [],
        "HarmonicBondForce": [],
        "HarmonicAngleForce": [],
        "RBTorsionForce": [],
        "NonbondedForce": {
            "coulomb14scale": "",
            "lj14scale": "",
            "Atoms": []
        }
    }

    # Collect AtomTypes
    for atom_type in root.find('AtomTypes'):
        data["AtomTypes"].append({
            "name": atom_type.get('name'),
            "class": atom_type.get('class'),
            "element": atom_type.get('element'),
            "mass": atom_type.get('mass'),
            "definition": atom_type.get('def'),
            "description": atom_type.get('desc'),
        })

    # Collect HarmonicBondForce
    for bond in root.find('HarmonicBondForce'):
        data["HarmonicBondForce"].append({
            "class1": bond.get('class1'),
            "class2": bond.get('class2'),
            "length": bond.get('length'),
            "force_constant": bond.get('k'),
        })

    # Collect HarmonicAngleForce
    for angle in root.find('HarmonicAngleForce'):
        data["HarmonicAngleForce"].append({
            "class1": angle.get('class1'),
            "class2": angle.get('class2'),
            "class3": angle.get('class3'),
            "angle": angle.get('angle'),
            "force_constant": angle.get('k'),
        })

    # Collect RBTorsionForce
    for torsion in root.find('RBTorsionForce'):
        data["RBTorsionForce"].append({
            "class1": torsion.get('class1'),
            "class2": torsion.get('class2'),
            "class3": torsion.get('class3'),
            "class4": torsion.get('class4'),
            "coefficients": [torsion.get(f'c{i}') for i in range(6)],
        })

    # Collect NonbondedForce attributes and atoms
    nonbonded = root.find('NonbondedForce')
    data["NonbondedForce"]["coulomb14scale"] = nonbonded.get('coulomb14scale')
    data["NonbondedForce"]["lj14scale"] = nonbonded.get('lj14scale')

    for atom in nonbonded:
        data["NonbondedForce"]["Atoms"].append({
            "type": atom.get('type'),
            "charge": atom.get('charge'),
            "sigma": atom.get('sigma'),
            "epsilon": atom.get('epsilon')
        })

    return data


def main():
    ethanol = load_pdb("ethanol.pdb")
    ethanol = strip_hydrogen_on_carbons(ethanol)
    parameters = load_parameters("ethanol.xml")
    pprint(ethanol)
    pprint(parameters)


if __name__ == "__main__":
    main()
