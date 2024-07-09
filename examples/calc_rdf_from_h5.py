""" Minimal example for calculing rdf from existing data """

import rumdpy as rp
import numpy as np
import os

if "LJ_T0.70.h5" not in os.listdir(os.getcwd()):
    print("This example needs a the file LJ_T0.70.h5 to be present")
    print("LJ_T0.70.h5 can be generated using minimal.py example")
    exit()

# Create configuration object
configuration = rp.Configuration()
# Load existing data
output = rp.tools.load_output("LJ_T0.70.h5")
# Read number of particles N and dimensions from data
nblocks, nconfs, _ , N, D = output['block'].shape
# Set up the configuration object
configuration['r'] = output['block'][0][0][0]
configuration.simbox = rp.Simbox(D, output['attrs']['simbox_initial'])
configuration.copy_to_device()
# Call the rdf calculator
calc_rdf = rp.CalculatorRadialDistribution(configuration, num_bins=1000)

# NOTE: the structure of the block is (outer_block, inner_steps, pos&img, npart, dimensions)
#       the zero is to select the position array and discard images
positions = output['block'][:,:,0,:,:]
positions = positions.reshape(nblocks*nconfs,N,D)
# Loop over saved configurations
for pos in positions[nconfs-1::nconfs]:
    configuration.set_vector('r', pos)
    configuration.copy_to_device()
    calc_rdf.update()

# Save rdf
calc_rdf.save_average()

