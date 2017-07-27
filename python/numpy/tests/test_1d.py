import numpy as np
from weldarray import WeldArray
import py.test
import random

UNARY_OPS = [np.exp, np.log, np.sqrt]
# TODO: Add wa.erf.
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
# TODO: Add int32, int64.
TYPES = ['float32', 'float64']
# TODO: Create test with all other ufuncs.

def random_arrays(num, dtype):
    '''
    Generates random Weld array, and numpy array of the given num elements.
    FIXME: Use np.random to generate random numbers of the given dtype.
    '''
    random.seed(1234)
    test = []
    for i in range(num):
        test.append(float(random.random()))
    test = np.array(test, dtype=dtype)

    # FIXME: Should be the same result as above, but this seems to fail for
    # some reason...
    # test = np.zeros((num), dtype=dtype)
    # test[:] = np.random.randn(*test.shape)

    np_test = np.copy(test)
    w = WeldArray(test)

    return np_test, w

def given_arrays(l, dtype):
    '''
    @l: list.
    returns a np array and a weldarray.
    '''
    test = np.array(l, dtype=dtype)
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
    np_test, w = random_arrays(10, 'float32')
    w = WeldArray(w)        # creating array again.
    w2 = np.exp(w)
    weld_result = w2.eval()
    np_result = np.exp(np_test)

    assert np.allclose(weld_result, np_result)

def test_array_indexing():
    '''
    Need to decide: If a weldarray item is accessed - should we evaluate the
    whole array (for expected behaviour to match numpy) or not?
    '''
    pass

def test_numpy_operations():
    '''
    Test operations that aren't implemented yet - it should pass it on to
    numpy's implementation, and return WeldArrays.
    '''
    np_test, w = random_arrays(10, 'float32')
    np_result = np.sin(np_test)
    w2 = np.sin(w)
    weld_result = w2.eval()

    assert np.allclose(weld_result, np_result)

def test_type_conversion():
    '''
    After evaluation, the dtype of the returned array must be the same as
    before.
    FIXME: Will need to generalize this after adding support for all types.
    '''
    _, w = random_arrays(10, 'float32')
    w2 = np.exp(w)
    weld_result = w2.eval()

    assert weld_result.dtype == 'float32'

def test_concat():
    '''
    Test concatenation of arrays - either Weld - Weld, or Weld - Numpy etc.
    '''
    pass

def test_views():
    '''
    Taking views into a 1d WeldArray should work as expected.
    In particular, this requires part of the initialization of the WeldArray be
    done in __array_finalize__ instead of __new__.

    FIXME: After supporting views, need to add more tests.
    '''
    np_test, w = random_arrays(10, 'float32')
    w = w[2:5]
    np_test = np_test[2:5]
    assert isinstance(w, WeldArray)
    w = np.exp(w)
    weld_result = w.eval()
    np_result = np.exp(np_test)

    assert np.allclose(weld_result, np_result)

def test_mix_np_weld_ops():
    '''
    Weld Ops + Numpy Ops - before executing any of the numpy ops, the
    registered weld ops must be evaluated.
    '''
    np_test, w = random_arrays(10, 'float32')
    np_test = np.exp(np_test)
    np_result = np.sin(np_test)

    w2 = np.exp(w)
    w2 = np.sin(w2)
    weld_result = w2.eval()
    assert np.allclose(weld_result, np_result)

def test_add_scalars():
    '''
    FIXME: Add all types of numbers.
    Special case of broadcasting rules - the scalar is applied to all the
    Weldrray members.
    '''
    np_test, w = random_arrays(10, 'float32')
    np_result = np_test + 2.00
    w2 = w + 2.00
    weld_result = w2.eval()
    assert np.allclose(weld_result, np_result)

def test_stale_add():
    '''
    Registers op for WeldArray w2, and then add it to w1. Works trivially
    because updating a weldobject with another weldobject just needs to get the
    naming right.
    '''
    n1, w1 = random_arrays(10, 'float32')
    n2, w2 = random_arrays(10, 'float32')

    w2 = np.exp(w2)
    n2 = np.exp(n2)

    w1 = np.add(w1, w2)
    n1 = np.add(n1, n2)

    w1 = w1.eval()
    assert np.allclose(w1, n1)

def test_cycle():
    '''
    the problem here seems to be reassiging to the same array.
    '''
    n1, w1 = given_arrays([1.0, 2.0], 'float32')
    n2, w2 = given_arrays([3.0, 3.0], 'float32')

    # w3 depends on w1.
    w3 = np.add(w1, w2)
    n3 = np.add(n1, n2)

    # changing this to some other variable lets us pass the test.
    w1 = np.exp(w1)
    n1 = np.exp(n1)

    w1 = np.add(w1,w3)
    n1 = np.add(n1, n3)

    assert np.allclose(w1.eval(), n1)
    assert np.allclose(w3.eval(), n3)

def test_self_assignment():
    n1, w1 = given_arrays([1.0, 2.0], 'float32')
    n2, w2 = given_arrays([2.0, 1.0], 'float32')

    w1 = np.exp(w1)
    n1 = np.exp(n1)
    assert np.allclose(w1.eval(), n1)

    w1 = w1 + w2
    n1 = n1 + n2

    assert np.allclose(w1.eval(), n1)

def test_reuse_array():
    '''
    a = np.add(b,)
    Ensure that despite sharing underlying memory of ndarrays, future ops on a
    and b should not affect each other as calculations are performed based on
    the weldobject which isn't shared between the two.
    '''
    n1, w1 = given_arrays([1.0, 2.0], 'float32')
    n2, w2 = given_arrays([2.0, 1.0], 'float32')

    w3 = np.add(w1, w2)
    n3 = np.add(n1, n2)

    w1 = np.log(w1)
    n1 = np.log(n1)

    w3 = np.exp(w3)
    n3 = np.exp(n3)

    w1 = w1 + w3
    n1 = n1 + n3

    w1_result = w1.eval()
    assert np.allclose(w1_result, n1)

    w3_result = w3.eval()
    assert np.allclose(w3_result, n3)
