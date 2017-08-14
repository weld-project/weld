import numpy as np
from weld.types import *

def is_view_child(child, par):
    '''
    Checks the base address of the given arrays to figure out if child and par
    have overlapping memory regions.
    '''
    return child.base is par

def addr(arr):
    '''
    returns address of the given ndarray.
    '''
    return arr.__array_interface__['data'][0]

def get_supported_binary_ops():
    '''
    Returns a dictionary of the Weld supported binary ops, with values being
    their Weld symbol.
    '''
    binary_ops = {}
    binary_ops[np.add.__name__] = '+'
    binary_ops[np.subtract.__name__] = '-'
    binary_ops[np.multiply.__name__] = '*'
    binary_ops[np.divide.__name__] = '/'
    return binary_ops

def get_supported_unary_ops():
    '''
    Returns a dictionary of the Weld supported unary ops, with values being
    their Weld symbol.
    '''
    unary_ops = {}
    unary_ops[np.exp.__name__] = 'exp'
    unary_ops[np.log.__name__] = 'log'
    unary_ops[np.sqrt.__name__] = 'sqrt'
    return unary_ops

def get_supported_types():
    '''
    '''
    types = {}
    types['float32'] = WeldFloat()
    types['float64'] = WeldDouble()
    types['int32'] = WeldInt()
    types['int64'] = WeldLong()
    return types

def get_supported_suffixes():
    '''
    Right now weld supports int32, int64, float32, float64.
    Treating python int as i32, float as f32
    '''
    suffixes = {}
    suffixes[str(np.int32)] = ''
    suffixes[str(int)] = ''
    suffixes[str(np.int64)] = 'L'
    suffixes[str(np.float32)] = 'f'
    suffixes[str(float)] = 'f'
    suffixes[str(np.float64)] = ''
    return suffixes

# TODO: turn these all into classes which provide functions.
# Global variables for the WeldArray type, used for lookups
BINARY_OPS = get_supported_binary_ops()
UNARY_OPS = get_supported_unary_ops()
SUPPORTED_DTYPES = get_supported_types()
DTYPE_SUFFIXES = get_supported_suffixes()
