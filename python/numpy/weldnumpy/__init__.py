from weldarray import *
from weldnumpy import *

# importing everything from numpy so we can selectively over-ride the array creation routines, and
# let other functions go to numpy.
from numpy import *

# since random in a module within numpy, we define a module that imports everything from random and
# returns weldarrays instead of ndarrays.
import weldrandom as random
import numpy as np

def array(arr, *args, **kwargs):
    '''
    Wrapper around weldarray - first create np.array and then convert to
    weldarray.
    '''
    return weldarray(np.array(arr, *args, **kwargs))

# TODO: define other array creation routines like zeros, ones etc.
