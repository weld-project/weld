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

# functions that don't exist in numpy
def erf(weldarray):
    '''
    FIXME: This is kinda hacky because all other function are routed through __array_ufunc__ by
    numpy and here we directly call _unary_op. In __array_ufun__ I was using properties of ufuncs,
    like ufunc.__name__, so using that route would require special casing stuff. For now, this is
    just the minimal case to make blackscholes work.
    '''
    return weldarray._unary_op('erf')
