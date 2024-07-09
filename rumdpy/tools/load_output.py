# This helper function is loading the output of a simulation as a dictionary following the formatting of sim.output
import sys
import numpy as np

# This function is a wrapper for several possible inputs
def load_output(name : str) -> dict:
    '''This function takes a simulation output and returns it as a dictionary in python following the formatting of sim.output'''
    if name[-3:]==".h5":
        print("Found .h5 file")
        return load_h5(name)
    elif name=="TrajectoryFiles":
        print("Found TrajectoryFiles")
        return load_rumd3(name)
    else:
        print("Input not recognized")
        return

# Load from h5 (std rumpdy output)
def load_h5(name:str) -> dict:
    import h5py
    output = dict()
    with h5py.File(name, "r") as f:
        # Import simulation data from .h5
        output = {i : np.array(f[i]) for i in f.parent.keys()}
        # Import metadata from h5
        # NOTE: .h5 has "steps_between_output" in f.attrs while sim.output['attrs'] has no that item. 
        # "steps_between_output" is a separate entry of sim.output
        output['attrs'] = {i: f.attrs.get(i) for i in f.attrs.keys()}
        output['steps_between_output'] = output['attrs'].pop('steps_between_output')
    return output

# Load from TrajectoryFiles (std rumd3 output)
def load_rumd3(name:str) -> dict:
    import gzip
    return

if __name__ == '__main__':
    load_output(sys.argv[1])
