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

'''
Transformation routines:
    1. transpose
    2. vstack
    3. hstack
'''
def transpose(*args, **kwargs):
    input1 = args[0]
    w = np.transpose(*args, **kwargs)
    # create the weldarray view.
    if isinstance(input1, weldarray):
        w = input1._gen_weldview(w)
    return w

# functions that don't exist in numpy
def erf(weldarray):
    '''
    FIXME: This is kinda hacky because all other function are routed through __array_ufunc__ by
    numpy and here we directly call _unary_op. In __array_ufun__ I was using properties of ufuncs,
    like ufunc.__name__, so using that route would require special casing stuff. For now, this is
    just the minimal case to make blackscholes work.
    '''
    return weldarray._unary_op('erf')
