import numpy as np

# Can be doctested: https://docs.python.org/3/library/doctest.html
# python3 -m doctest    colarray.py  # No output == No problem!
# python3 -m doctest -v colarray.py  # The verbose version

class colarray():
    """ The Column array Class

    A class storing several sets ('columns') of lengths with identical dimensions in a single numpy array. Strings are used as indicies along the zeroth dimension corresponding to different columns of lengths.
    
    Examples
    --------

    Storage for positions, velocities, and forces, for 1000 particles in 2 dimensions:

    >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
    >>> ca.shape
    (3, 1000, 2)
    >>> ca.column_names
    ('r', 'v', 'f')
    Data is accesed via string indicies (similar to dataframes in pandas):
    >>> ca['r'] = np.ones((1000,2))
    >>> ca['v'] = 2   # Broadcastet by numpy to correct shape
    >>> print(ca['r'] + 0.01*ca['v'])
    [[1.02 1.02]
     [1.02 1.02]
     [1.02 1.02]
     ...
     [1.02 1.02]
     [1.02 1.02]
     [1.02 1.02]]
    
    Assignment have to use an existing key (as opposed to eg. pandas):
    
    >>> ca['c'] = 1
    Traceback (most recent call last):
        ...
    KeyError: 'c'
    
    On assignment the right hand side needs to be a numpy array compatible with size of columns originaly specified (possibly after broadcasting by numpy):
    
    >>> ca['f'] =  np.ones((100,2))
    Traceback (most recent call last):
        ...
    ValueError: could not broadcast input array from shape (100,2) into shape (1000,2)

    To assign indicies to variable names (say to use on a GPU):
    
    >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
    >>> for col in ca.column_names:
    ...    exec(f'ca_{col}_id = {ca.indicies[col]}', globals())
    >>> print(ca_r_id, ca_v_id, ca_f_id)
    0 1 2
    
    """
    
    # Most error handling is left to be handled by numpy, as it gives usefull error messages 
    # (illustrated in the documentation string above).
    
    def __init__(self, column_names, size, dtype=np.float32, array=None):
        self.column_names = column_names
        self.dtype = dtype
        self.indicies = {key:index for index,key in enumerate(column_names)}
        if type(array)==np.ndarray: # Used, e.g.,  when loading from file
            self.array = array
        else:
            self.array = np.zeros((len(column_names), *size), dtype=dtype) 
            
        self.shape = self.array.shape

    def __setitem__(self, key, data):
        self.array[self.indicies[key]] = data
        
    def __getitem__(self, key):
        return self.array[self.indicies[key]]
   
    def __repr__(self):
        return 'colarray('+str(tuple(self.indicies.keys()))+', '+self.array.shape[1:].__repr__()+')\n'+self.array.__repr__()
    
    def copy(self):
        return colarray(self.column_names, self.shape, self.dtype, self.array.copy())


def save(file, colarray):
    """
    Save a colarray to disk.
    >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
    >>> save('test_colarray', ca)
    """
    np.save(f'{file}.npy', colarray.array)
    with open(f'{file}.col', 'w') as f:    # Use pickle / json
        f.write(str(len(colarray.column_names)) + '\n')
        for key in colarray.column_names:
            f.write(key + '\n')
    return


def load(file):
    """
    Load a colarray from disk.
    >>> ca = colarray(('r', 'v', 'f'), size=(1000,2))
    >>> ca['f'] = np.random.uniform(size=(1000,2))    
    >>> save('test_colarray', ca)
    >>> ca2 = load('test_colarray')
    >>> for col in ca.column_names:
    ...     print(np.all(ca2[col]==ca[col]))
    True
    True
    True
    
    The file(s) needs to be present:
    >>> ca2 = load('test_colarray_not_there')
    Traceback (most recent call last):
        ...
    FileNotFoundError: [Errno 2] No such file or directory: 'test_colarray_not_there.col'
    """
    
    with open(f'{file}.col', 'r') as f:
        num_columns = int(f.readline())
        column_names = []
        for i in range(num_columns):
            column_names.append(f.readline()[:-1]) # removing '\n'
    array = np.load(f'{file}.npy')
    return colarray(column_names, array.shape[1:], array=array)


