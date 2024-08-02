""" Minimal example for calculing rdf from existing data """

import os

import rumdpy as rp

file_to_read = "TrajectoryFiles"

if file_to_read not in os.listdir(os.getcwd()):
    print(f"This example needs {file_to_read} to be present")
    exit()

# Load existing data
output = rp.tools.load_output(file_to_read)
# Read number of particles N and dimensions from data
nblocks, nconfs, _ , N, D = output['block'].shape
# Set up the configuration object
configuration = rp.Configuration(D=D, N=N)
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
    configuration['r'] = pos
    configuration.copy_to_device()
    calc_rdf.update()

# Save rdf
calc_rdf.save_average("rdf_rumd3.dat")
