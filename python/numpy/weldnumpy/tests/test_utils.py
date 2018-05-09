import numpy as np
from weldnumpy import *

# TODO: combine unary and trig ops.
UNARY_OPS = [np.exp, np.log, np.sqrt]
TRIG_OPS = [np.sin, np.cos]
# TODO: Add wa.erf - doesn't use the ufunc functionality of numpy so not doing it for
# now.
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
REDUCE_UFUNCS = [np.add.reduce, np.multiply.reduce]
CMP_OPS = [np.less, np.less_equal, np.greater, np.greater_equal]

TYPES = ['float32', 'float64', 'int32', 'int64']
SHAPES = [10, (2,2), (3,7), (9,1,4), (2,5,7,2)]
NUM_ELS = 10

# TODO: Create test with all other ufuncs.
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
    w = weldarray(test, verbose=False)

    return np_test, w

def given_arrays(l, dtype):
    '''
    @l: list.
    returns a np array and a weldarray.
    '''
    test = np.array(l, dtype=dtype)
    np_test = np.copy(test)
    w = weldarray(test)

    return np_test, w

