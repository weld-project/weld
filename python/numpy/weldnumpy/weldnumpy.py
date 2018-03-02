import numpy as np
import scipy.special as ss
from weld.types import *
from distutils.version import StrictVersion
assert StrictVersion(np.__version__) >= StrictVersion('1.13')

ALL_PASSES = ["loop-fusion", "infer-size", "short-circuit-booleans",
        "predicate", "vectorize", "fix-iterate"]
CUR_PASSES = ALL_PASSES
offload_setitem = True

class weldarray_view():
    '''
    This can be either a parent or a child.
    '''
    def __init__(self, base_array, parent, start, end, idx, shape=None,
            strides=None):
        '''
        TODO: Need to add more stuff / generalize to nd case.
        TODO: Can also just use w.base instead of storing base_array here.
        @base_array: base array from which all views are derived.
        @parent: direct parent from which this view is generated.
        @start:  starting index of view in base_array.
        @end:    ending index of view in base_array.
        @idx:    the slicing notation used to generate the view.
        '''
        self.base_array = base_array
        self.parent = parent
        # TODO: should we try to separate 1d views with multi-d views?
        self.start = start
        self.end = end
        self.idx = idx

        # these are used exclusively for nditer -- and non-contiguous arrays.
        self.shape = shape
        self.strides = strides

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
    unary_ops[np.sin.__name__] = 'sin'
    unary_ops[np.cos.__name__] = 'cos'
    unary_ops[np.tan.__name__] = 'tan'
    unary_ops[np.arccos.__name__] = 'acos'
    unary_ops[np.arcsin.__name__] = 'asin'
    unary_ops[np.arctan.__name__] = 'atan' 
    unary_ops[np.sinh.__name__] = 'sinh'
    unary_ops[np.cosh.__name__] = 'cosh'
    unary_ops[np.tanh.__name__] = 'thanh'
    # scipy functions also seem to be routed through ufuncs!
    unary_ops[ss.erf.__name__] = 'erf'
    return unary_ops

def get_supported_cmp_ops():
    cmp_ops = {}
    cmp_ops[np.less_equal.__name__] = '<='
    cmp_ops[np.less.__name__] = '<'
    cmp_ops[np.greater_equal.__name__] = '>='
    cmp_ops[np.greater.__name__] = '>'
    return cmp_ops

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
CMP_OPS = get_supported_cmp_ops()
SUPPORTED_DTYPES = get_supported_types()
DTYPE_SUFFIXES = get_supported_suffixes()


# XXX: FIXME get rid of this.
def random_arrays(shape, dtype):
    '''
    Generates random Weld array, and numpy array of the given num elements.
    '''
    # np.random does not support specifying dtype, so this is a weird
    # way to support both float/int random numbers
    test = np.zeros((shape), dtype=dtype)
    test[:] = np.random.randn(*test.shape)
    test = np.abs(test)
    # at least add 1 so no 0's (o.w. divide errors)
    random_add = np.random.randint(1, high=10, size=test.shape)
    test = test + random_add
    test = test.astype(dtype)

    np_test = np.copy(test)
    # w = weldarray(test, verbose=False)

    return np_test

def remove_pass(pass_name):
    global CUR_PASSES
    for l in CUR_PASSES:
        if pass_name in l:
            CUR_PASSES.remove(l)
            break
# Stuff for the incremental integration study.
def remove_all_ops():
    global BINARY_OPS, UNARY_OPS, CMP_OPS
    BINARY_OPS = {}
    UNARY_OPS = {}
    CMP_OPS = {}

def add_ops(ops):
    '''
    @ops: list of strings, where string would be the op.__name__ of the numpy op.
    '''
    global BINARY_OPS, UNARY_OPS, CMP_OPS
    unary_ops = get_supported_unary_ops()
    binary_ops = get_supported_binary_ops()
    cmp_ops = get_supported_cmp_ops()
    for op in ops:
        if op in binary_ops:
            BINARY_OPS[op] = binary_ops[op]
        elif op in unary_ops:
            UNARY_OPS[op] = unary_ops[op]
        elif op in cmp_ops:
            CMP_OPS[op] = cmp_ops[op]

def set_offload_setitem(val):
    global offload_setitem
    offload_setitem = val
