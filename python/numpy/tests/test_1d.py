import numpy as np
from weldarray import WeldArray
import py.test
import random

UNARY_OPS = [np.exp, np.log, np.sqrt]
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
TYPES = ["float32", "float64"]

def random_arrays(num, dtype):
    '''
    Generates random Weld array, and numpy array of the given num elements.
    FIXME: Use np.random to generate random numbers of the given dtype.
    '''
    random.seed(1234)
    test = []
    for i in range(num):
        test.append(float(random.random()))

    # FIXME: Should be the same result as above, but this seems to fail for
    # some reason...
    # test = np.zeros((num), dtype=dtype)
    # print(test)
    # test[:] = np.random.randn(*test.shape)
    # print(test)

    test = np.array(test, dtype=dtype)
    np_test = np.copy(test)
    w = WeldArray(test)

    return np_test, w

def test_unary_elemwise():
    '''
    Tests all the unary ops in UNARY_OPS.

    TODO: Add tests for all supported datatypes
    TODO: Factor out code for binary tests.
    '''
    for op in UNARY_OPS:
        for dtype in TYPES:
            np_test, w = random_arrays(10, dtype)
            w2 = op(w)
            weld_result = w2.eval()
            np_result = op(np_test)
            assert np.allclose(weld_result, np_result)

def test_binary_elemwise():
    '''
    '''
    for op in BINARY_OPS:
        for dtype in TYPES:
            np_test, w = random_arrays(10, dtype)
            np_test2, w2 = random_arrays(10, dtype)
            w3 = op(w, w2)
            weld_result = w3.eval()
            np_result = op(np_test, np_test2)

            assert np.allclose(weld_result, np_result)

def test_multiple_array_creation():
    '''
    Minor edge case but it fails right now.
    ---would probably be fixed after we get rid of the loop fusion at the numpy
    level.
    '''
    np_test, w = random_arrays(10, "float32")
    w = WeldArray(w)        # creating array again.
    w2 = np.exp(w)
    weld_result = w2.eval()
    np_result = np.exp(np_test)

    assert np.allclose(weld_result, np_result)

def test_array_indexing():
    '''
    Whenever WeldArrays are indexed, they should be evaluated to the current
    values (rather than just tracking the registered computations...)
    '''
    pass

def test_numpy_operations():
    '''
    Test operations that aren't implemented yet - it should pass it on to
    numpy's implementation, and return WeldArrays.
    '''
    np_test, w = random_arrays(10, "float32")
    np_result = np.sin(np_test)
    w2 = np.sin(w)
    weld_result = w2.eval()

    assert np.allclose(weld_result, np_result)

def test_type_conversion():
    '''
    After evaluation, the dtype of the returned array must be the same as
    before. Not the case in many situations...
    FIXME: Will need to generalize this after adding support for all types.
    '''
    _, w = random_arrays(10, "float32")
    w2 = np.exp(w)
    weld_result = w2.eval()

    assert weld_result.dtype == "float32"

def test_concat():
    '''
    Test concatenation of arrays - either Weld - Weld, or Weld - Numpy etc.
    '''
    pass

def test_views():
    '''
    Taking views into a 1d WeldArray should work as expected.
    In particular, this requires all the initialization of the WeldArray be
    done in __array_finalize__ instead of __new__.
    '''
    np_test, w = random_arrays(10, "float32")
    w = w[2:5]
    np_test = np_test[2:5]
    assert isinstance(w, WeldArray)
    w = np.exp(w)
    weld_result = w.eval()
    np_result = np.exp(np_test)

    assert np.allclose(weld_result, np_result)

def test_mix_np_weld_ops():
    '''
    '''
    np_test, w = random_arrays(10, "float32")
    np_test = np.exp(np_test)
    np_result = np.sin(np_test)

    w2 = np.exp(w)
    w2 = np.sin(w2)
    weld_result = w2.eval()
    assert np.allclose(weld_result, np_result)

def test_add_scalars():

    np_test, w = random_arrays(10, "float32")
    np_result = np_test + 2.00
    w2 = w + 2.00
    weld_result = w2.eval()
    assert np.allclose(weld_result, np_result)


