"""This script creates a new python script in which a diction containing the
parameters for the ZJW-2004 potential are contained. These are the same
parameters that were published in the 2004 paper. They are read in from the
text file  'ZJW-2004-parameters.txt'

"""
import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py
import os

# to allow this script to be run by testing need to specify that the location
# of theinput file is the same as that of the script itself
script_dir = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(script_dir, 'ZJW-2004-parameters.txt')

with open(filename, 'r') as f:
    header_line = f.readline().strip()

column_names = header_line.lstrip('#').split()

df = pd.read_csv(filename, sep='\s+', skiprows=1, names=column_names)


EAM_ZJW_2004_params = {}

for element in column_names:

    EAM_ZJW_2004_params[element] = df[element].to_numpy()

outfilename = "EAM_ZJW_2004_params_TMP.py"

with open(outfilename, "w") as f:
    f.write("from numpy import array\n")
    print("EAM_ZJW_2004_params = ", EAM_ZJW_2004_params, file=f)

print("Wrote parameters as a dictionary to", outfilename)
print("Copy to EAM_ZJW_2004_params.py if necessary")
