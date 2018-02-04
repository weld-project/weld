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

def zeros(*args, **kwargs):
    return weldarray(np.zeros(*args, **kwargs))

def zeros_like(*args, **kwargs):
    return weldarray(np.zeros_like(*args, **kwargs))

def ones(*args, **kwargs):
    return weldarray(np.ones(*args, **kwargs))

def ones_like(*args, **kwargs):
    return weldarray(np.ones_like(*args, **kwargs))

def full(*args, **kwargs):
    return weldarray(np.full(*args, **kwargs))

def full_like(*args, **kwargs):
    return weldarray(np.full_like(*args, **kwargs))

def empty(*args, **kwargs):
    return weldarray(np.empty(*args, **kwargs))

def empty_like(*args, **kwargs):
    return weldarray(np.empty_like(*args, **kwargs))

def eye(*args, **kwargs):
    return weldarray(np.eye(*args, **kwargs))

def identity(*args, **kwargs):
    return weldarray(np.identity(*args, **kwargs))
