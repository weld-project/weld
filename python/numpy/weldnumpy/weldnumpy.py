import numpy as np
from weld.types import *

class weldarray_view():
    '''
    This can be either a parent or a child.
    '''
    def __init__(self, base_array, parent, idx):
        '''
        TODO: Can also just use w.base instead of storing base_array here.
        @base_array: base array from which all views are derived.
        @parent: direct parent from which this view is generated. It is convenient to store this
        for evaluation.
        @segments: list of segments (of the parent array) that belong to this view.
        @idx: the slicing notation used to generate the view (from the parent) - for convenience in
        _eval.
        '''
        self.base_array = base_array
        self.parent = parent
        self.idx = idx
        # FIXME: need to have an absolute start for this welview in the parent.  does not need to
        # be here! Can just be a separate field in the weldarray, because not only views would have
        # it.

class segment():
    '''
    represents one segment of a view
    '''
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

def is_view_child(view, par):
    '''
    Checks the base address of the given arrays to figure out if child and par
    have overlapping memory regions.
    '''
    if par.base is None:
        # par is the base array.
        return view.base is par
    else:
        # par is a view of another array as well!
        # view can only be a child of par if they share base.
        return view.base is par.base

def addr(arr):
    '''
    returns raw address of the given ndarray.
    '''
    return arr.__array_interface__['data'][0]


def get_supported_binary_ops():
    '''
    Returns a dictionary of the Weld supported binary ops, with values being their Weld symbol.
    '''
    binary_ops = {}
    binary_ops[np.add.__name__] = '+'
    binary_ops[np.subtract.__name__] = '-'
    binary_ops[np.multiply.__name__] = '*'
    binary_ops[np.divide.__name__] = '/'
    return binary_ops

def get_supported_unary_ops():
    '''
    Returns a dictionary of the Weld supported unary ops, with values being their Weld symbol.
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
    '''
    suffixes = {}
    suffixes['i32'] = ''
    suffixes['i64'] = 'L'
    suffixes['f32'] = 'f'
    suffixes['f64'] = ''

    return suffixes

# TODO: turn these all into classes which provide functions.
# Global variables for the WeldArray type, used for lookups
BINARY_OPS = get_supported_binary_ops()
UNARY_OPS = get_supported_unary_ops()
SUPPORTED_DTYPES = get_supported_types()
DTYPE_SUFFIXES = get_supported_suffixes()
