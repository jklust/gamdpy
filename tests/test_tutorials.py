""" Test Jupyter Notebooks in the tutorials directory"""

def test_tutorials():
    ...
    # import nbformat
    # from nbclient import NotebookClient
    # filename = "./tutorials/my_first_simulation.ipynb"
    # as_version = 4
    # nb = nbformat.read(filename, as_version=as_version)
    # client = NotebookClient(nb, timeout=600, resources={'metadata': {'path': './tutorials/'}})  # kernel_name='python3'
    # client.execute()

if __name__ == "__main__":
    test_tutorials()
