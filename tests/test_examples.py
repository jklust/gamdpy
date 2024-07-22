""" Try to run all the scripts in the examples directory

The plt.show() function is replaced by a dummy function to avoid showing the plots.
This script will skip some examples that are known to fail, see variable exclude_files.
When debugging, you can change variable files to a few or a single file.
"""
import time
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Agg')  # Static backend that does not halt on plt.show()

def test_examples(path_to_examples='examples'):

    # List of scripts to exclude
    exclude_files = [
        'switching_integrator.py',  # FileNotFoundError: [Errno 2] No such file or directory: 'Data/isomorph.pkl'
        'plot_isomorph_rdf.py',  # FileNotFoundError: [Errno 2] No such file or directory: 'Data/isomorph.pkl'
        'test_shear.py',
        # FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'LJ_cooled_0.70.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
        'LJchain_wall.py',  # ImportError: cannot import name 'nvt_nh' from 'rumdpy.integrators'
        'yukawa.py',  # NameError: name 'yukawa' is not defined
        'calc_rdf_from_rumd3.py',  # SystemExit: None
        'thermodynamics.py',  # NameError: name 'error_estimate_by_blocking' is not defined
        'LJchain.py',  # NameError: name 'np' is not defined
        'minimal_cpu.py',  # I suspect this script makes other scripts fail due to the os.environ[...] lines
        'consistency_NPT.py',  # Very slow: Execution time for consistency_NPT.py: 8.98e+02 s
        'ASD.py',  # Slow: Execution time for ASD.py: 56.3 s
    ]

    # Save the current working directory
    original_cwd = os.getcwd()
    examples_dir = os.path.abspath(path_to_examples)

    try:
        os.chdir(examples_dir)

        # Iterate over all Python files in the examples directory
        files = list(glob.glob('*.py'))
        files.sort()
        # files = ['minimal.py']  # Uncomment and modify for debugging a few or a single file
        print(f"Running {len(files)} examples: {files}")
        for file in files:
            if os.path.basename(file) in exclude_files:
                print(f"Skipping {file} (warning: may fail)")
                continue

            with open(file) as example:
                print(f"Executing {file}")
                tic = time.perf_counter()
                exec(example.read())
                toc = time.perf_counter()
                print(f"Execution time for {file}: {toc-tic:.3} s")
            # Close all matplotlib figures
            plt.close('all')
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    test_examples()

