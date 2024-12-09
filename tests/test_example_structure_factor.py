""" This test ensure that the example works as intended """
import os
import matplotlib

def test_example_structure_factor():
    # Use the non-interactive Agg Matplotlib backend to avoid interactive graphical interface
    matplotlib.get_backend()
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg')

    # Run the example
    example_folder = 'examples'
    example_script = 'structure_factor.py'
    example_path = os.path.join(example_folder, example_script)
    with open(example_path, 'r') as script:
        exec(script.read(), globals())

if __name__ == '__main__':
    test_example_structure_factor()
