""" Minimal example for calculing rdf from existing data """

import os

import rumdpy as rp

file_to_read = "LJ_T0.70.h5"

if file_to_read not in os.listdir(os.getcwd()):
    print("This example needs a the file LJ_T0.70.h5 to be present")
    print(f"{file_to_read} can be generated using minimal.py example")
    exit()

# Load existing data
output = rp.tools.load_output(file_to_read)

# Read number of particles N and dimensions from data
nblocks, nconfs, _ , N, D = output['block'].shape

# Set up the configuration object
# Create configuration object
configuration = rp.Configuration(D=D, N=N)
configuration.simbox = rp.Simbox(D, output['attrs']['simbox_initial'])
configuration.copy_to_device()

# Call the rdf calculator
calc_sq = rp.CalculatorStructureFactor(configuration)
calc_sq.generate_q_vectors(q_max=18)

# NOTE: the structure of the block is (outer_block, inner_steps, pos&img, npart, dimensions)
#       the zero is to select the position array and discard images
positions = output['block'][:,:,0,:,:]
positions = positions.reshape(nblocks*nconfs,N,D)

# Loop over saved configurations
for pos in positions[nconfs-1::nconfs]:
    configuration['r'] = pos
    configuration.copy_to_device()
    calc_sq.update()

# Save sq
calc_sq.save_average()
# The if statements are to avoid pytest to write the file to disk)
if 'RUMDPY_SAVE_OUTPUT_EXAMPLES' in os.environ:
    if os.environ['RUMDPY_SAVE_OUTPUT_EXAMPLES']=='0': 
        os.remove('sq.dat')

