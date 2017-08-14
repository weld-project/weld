import numpy as np
import py.test
import random
from weldnumpy import *

UNARY_OPS = [np.exp, np.log, np.sqrt]
# TODO: Add wa.erf - doesn't use the ufunc functionality so not doing it for
# now.
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
TYPES = ['float32', 'float64', 'int32', 'int64']

NUM_ELS = 5

# TODO: Create test with all other ufuncs.
def random_arrays(num, dtype):
    '''
    Generates random Weld array, and numpy array of the given num elements.
    '''
    # np.random does not support specifying dtype, so this is a weird
    # way to support both float/int random numbers
    test = np.zeros((num), dtype=dtype)
    test[:] = np.random.randn(*test.shape)
    test = np.abs(test)
    # at least add 1 so no 0's (o.w. divide errors)
    random_add = np.random.randint(1, high=10, size=test.shape)
    test = test + random_add
    test = test.astype(dtype)

    np_test = np.copy(test)
    w = weldarray(test)

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

def test_unary_elemwise():
    '''
    Tests all the unary ops in UNARY_OPS.

    TODO: Add tests for all supported datatypes
    TODO: Factor out code for binary tests.
    TODO: For now, unary ops seem to only be supported on floats.
    '''
    for op in UNARY_OPS:
        for dtype in TYPES:
            # int still not supported for the unary ops in Weld.
            if "int" in dtype:
                continue
            np_test, w = random_arrays(NUM_ELS, dtype)
            w2 = op(w)
            print(w2)
            weld_result = w2.evaluate()
            np_result = op(np_test)
            assert np.allclose(weld_result, np_result)

def test_binary_elemwise():
    '''
    '''
    for op in BINARY_OPS:
        for dtype in TYPES:
            np_test, w = random_arrays(NUM_ELS, dtype)
            np_test2, w2 = random_arrays(NUM_ELS, dtype)
            print(w2)
            w3 = op(w, w2)
            weld_result = w3.evaluate()
            np_result = op(np_test, np_test2)
            # Need array equal to keep matching types for weldarray, otherwise
            # allclose tries to subtract floats from ints.
            assert np.array_equal(weld_result, np_result)

def test_multiple_array_creation():
    '''
    Minor edge case but it fails right now.
    ---would probably be fixed after we get rid of the loop fusion at the numpy
    level.
    '''
    np_test, w = random_arrays(NUM_ELS, 'float32')
    w = weldarray(w)        # creating array again.
    w2 = np.exp(w)
    weld_result = w2.evaluate()
    np_result = np.exp(np_test)

    assert np.allclose(weld_result, np_result)

def test_array_indexing():
    '''
    Need to decide: If a weldarray item is accessed - should we evaluateuate the
    whole array (for expected behaviour to match numpy) or not?
    '''
    pass

def test_numpy_operations():
    '''
    Test operations that aren't implemented yet - it should pass it on to
    numpy's implementation, and return weldarrays.
    '''
    np_test, w = random_arrays(NUM_ELS, 'float32')
    np_result = np.sin(np_test)
    w2 = np.sin(w)
    weld_result = w2.evaluate()

    assert np.allclose(weld_result, np_result)

def test_type_conversion():
    '''
    After evaluateuation, the dtype of the returned array must be the same as
    before.
    '''
    for t in TYPES:
        _, w = random_arrays(NUM_ELS, t)
        _, w2 = random_arrays(NUM_ELS, t)
        w2 = np.add(w, w2)
        weld_result = w2.evaluate()
        assert weld_result.dtype == t

def test_concat():
    '''
    Test concatenation of arrays - either Weld - Weld, or Weld - Numpy etc.
    '''
    pass

def test_views_basic():
    '''
    Taking views into a 1d weldarray should return a weldarray view of the
    correct data without any copying.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    w2 = w[2:5]
    n2 = n[2:5]
    assert isinstance(w2, weldarray)

def test_views_update_child():
    '''
    If the values of a given view are changed - then these changes should be
    reflected in the bigger parent array.
    FIXME: One way to solve this would be - if we register an op for the child
    array, then also register the same op for the equivalent elements of the
    parent array...but this would add a bunch of code which doesn't seem as
    clean.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    print("orig n: ", n)
    print("orig w: ", w)

    w2 = w[2:5]
    n2 = n[2:5]

    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    w2.evaluate()
    print("after exp, addr: ", addr(w2))

    print("n[2:5] = ", n[2:5])
    print("w[2:5] = ", w[2:5])
    print("w2: ", w2)
    print("n2: ", n2)

    print("before all close")
    assert np.allclose(w[2:5], w2)
    print("after all close")

    # assert np.allclose(w[2:5], w2.evaluate())

    # assert np.allclose(w2.evaluate(), n2)

def test_views_update_parent():
    '''
    Create a view, then update the parent in place. The change should be
    effected in the view-child as well.
    ''' 
    n, w = random_arrays(NUM_ELS, 'float32')
    w2 = w[2:4]
    n2 = n[2:4]

    w = np.exp(w, out=w)
    n = np.exp(n, out=n)

    # w2 should have been updated too.    
    assert np.allclose(n[2:4], n2)
    assert np.allclose(w[2:4], w2.evaluate())
    
    # Part 2: binops don't work for us yet.
    # n3, w3 = random_arrays(NUM_ELS, 'float32')
    # w = np.add(w, w3, out=w)
    # n = np.add(n, n3, out=n)

    # assert np.allclose(n[2:4], n2)
    # assert np.allclose(w[2:4], w2.evaluate())

def test_views_mess():
    '''
    More complicated versions of the views test.
    '''
    pass

def test_mix_np_weld_ops():
    '''
    Weld Ops + Numpy Ops - before executing any of the numpy ops, the
    registered weld ops must be evaluateuated.
    '''
    np_test, w = random_arrays(NUM_ELS, 'float32')
    np_test = np.exp(np_test)
    np_result = np.sin(np_test)

    w2 = np.exp(w)
    w2 = np.sin(w2)
    weld_result = w2.evaluate()
    assert np.allclose(weld_result, np_result)

def test_scalars():
    '''
    FIXME: Not sure how f64/i64 is represented in weld.
    Special case of broadcasting rules - the scalar is applied to all the
    Weldrray members.
    '''
    t = "int32"
    print("t = ", t)
    np_test, w = random_arrays(NUM_ELS, t)
    np_result = np_test + 2
    w2 = w + 2
    weld_result = w2.evaluate()
    assert np.allclose(weld_result, np_result)

    t = "float32"
    print("t = ", t)
    np_test, w = random_arrays(NUM_ELS, t)
    np_result = np_test + 2.00
    w2 = w + 2.00
    weld_result = w2.evaluate()
    assert np.allclose(weld_result, np_result)

def test_stale_add():
    '''
    Registers op for weldarray w2, and then add it to w1. Works trivially
    because updating a weldobject with another weldobject just needs to get the
    naming right.
    '''
    n1, w1 = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')

    w2 = np.exp(w2)
    n2 = np.exp(n2)

    w1 = np.add(w1, w2)
    n1 = np.add(n1, n2)

    w1 = w1.evaluate()
    assert np.allclose(w1, n1)

def test_cycle():
    '''
    This was a problem when I was using let statements to hold intermediate
    weld code. (because of my naming scheme)
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

    assert np.allclose(w1.evaluate(), n1)
    assert np.allclose(w3.evaluate(), n3)

def test_self_assignment():
    n1, w1 = given_arrays([1.0, 2.0], 'float32')
    n2, w2 = given_arrays([2.0, 1.0], 'float32')

    w1 = np.exp(w1)
    n1 = np.exp(n1)
    assert np.allclose(w1.evaluate(), n1)

    w1 = w1 + w2
    n1 = n1 + n2

    assert np.allclose(w1.evaluate(), n1)

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

    w1_result = w1.evaluate()
    assert np.allclose(w1_result, n1)

    w3_result = w3.evaluate()
    assert np.allclose(w3_result, n3)

def test_iterator():
    '''
    While iterating, the values of the np array must be the latest after
    evaluateuated stored ops.
    '''
    pass

def test_fancy_indexing():
    '''
    TODO: Needs more complicated tests that mix different indexing strategies,
    but since fancy indexing creates a new array - it shouldn't have any
    problems dealing with further stuff.
    '''
    _, w = random_arrays(NUM_ELS, 'float64')
    b = w > 0.50
    w2 = w[b]
    assert isinstance(w2, weldarray)

def test_mixing_types():
    '''
    mixing f32 with f64, or i32 with f64.
    Weld doesn't seem to support this right now, so pass it on to np.
    '''
    n1, w1 = random_arrays(2, 'float64')
    n2, w2 = random_arrays(2, 'float32')

    w3 = w1 + w2
    n3 = n1 + n2
    assert np.array_equal(n3, w3.evaluate())

def test_inplace_assignment():
    '''
    With the output optimization, this should be quite efficient for weld.
    '''
    n, w = random_arrays(100, 'float32')
    n2, w2 = random_arrays(100, 'float32')

    orig_addr = id(w)

    for i in range(100):
        n += n2
        w += w2

    # Ensures that the stuff above happened in place.
    assert id(w) == orig_addr
    w3 = w.evaluate()
    assert np.allclose(n, w)

def test_nested_weld_expr():
    '''
    map(zip(map(...))) kind of really long nested expressions.
    Add a timeout - it shouldn't take literally forever as it does now.
    '''
    pass

def test_getitem_evaluate():
    '''
    Should evaluateuate stuff before returning from getitem.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')

    n += n2
    w += w2

    assert n[0] == w[0]

def test_implicit_evaluate():
    n, w = random_arrays(2, 'float32')
    n2, w2 = random_arrays(2, 'float32')

    w3 = w+w2
    n3 = n+n2
    print(w3)
    w3 = w3.evaluate()
    w3 = w3.evaluate()

    assert np.allclose(w3, n3)

# test_views_update_parent()
# test_views_update_child()
