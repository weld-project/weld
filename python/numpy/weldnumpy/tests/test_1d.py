import numpy as np
import py.test
import random
from weldnumpy import weldarray
import scipy.special as ss

'''
TODO: New tests:
    - edge/failing cases: out = ndarray for op involving weldarrays.
'''

#UNARY_OPS = [np.exp, np.log, np.sqrt, np.square]
UNARY_OPS = [np.square, np.exp, np.log, np.sqrt]
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
REDUCE_UFUNCS = [np.add.reduce, np.multiply.reduce]
# TODO: add other similar functions
UTIL_FUNCTIONS = [np.max, np.min]

# FIXME: weld mergers dont support non-commutative ops --> need to find a workaround for this.
# REDUCE_UFUNCS = [np.add.reduce, np.subtract.reduce, np.multiply.reduce, np.divide.reduce]

TYPES = ['float32', 'float64', 'int32', 'int64']
NUM_ELS = 10

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

def test_unary_elemwise():
    '''
    Tests all the unary ops in UNARY_OPS.

    FIXME: For now, unary ops seem to only be supported on floats.
    '''
    for op in UNARY_OPS:
        for dtype in TYPES:
            # int still not supported for the unary ops in Weld.
            if "int" in dtype:
                continue
            np_test, w = random_arrays(NUM_ELS, dtype)
            w2 = op(w)
            np_result = op(np_test)
            w2_eval = w2.evaluate()

            assert np.allclose(w2_eval, np_result)
            assert np.array_equal(w2_eval, np_result)

def test_binary_elemwise():
    '''
    '''
    for op in BINARY_OPS:
        for dtype in TYPES:
            np_test, w = random_arrays(NUM_ELS, dtype)
            np_test2, w2 = random_arrays(NUM_ELS, dtype)
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
    After evaluating, the dtype of the returned array must be the same as
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
    Updates both parents and child to put more strain.
    '''
    def asserts(w, n, w2, n2):
        w = w.evaluate()
        assert np.allclose(w, n)
        assert np.allclose(w2.evaluate(), n2)

        assert np.allclose(w[2:5].evaluate(), w2.evaluate())

    NUM_ELS = 10
    n, w = random_arrays(NUM_ELS, 'float32')

    w2 = w[2:5]
    n2 = n[2:5]

    # unary part
    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    # the weldarray view object must be preserved at this point.
    w2 = w2.evaluate()
    asserts(w, n, w2, n2)
    # binary part
    n3, w3 = random_arrays(3, 'float32')

    n2 = np.add(n2, n3, out=n2)
    w2 = np.add(w2, w3, out=w2)
    w2 = w2.evaluate()
    asserts(w, n, w2, n2)

    w2 += 5.0
    n2 += 5.0
    w2 = w2.evaluate()

    asserts(w, n, w2, n2)

def test_views_update_parent():
    '''
    Create a view, then update the parent in place. The change should be
    effected in the view-child as well.
    '''
    def asserts(w, n, w2, n2):
        assert np.allclose(w, n)
        assert np.allclose(w[2:4].evaluate(), w2.evaluate())
        assert np.allclose(w2.evaluate(), n2)

    n, w = random_arrays(NUM_ELS, 'float32')
    w2 = w[2:4]
    n2 = n[2:4]

    w = np.exp(w, out=w)
    n = np.exp(n, out=n)
    w2.evaluate()

    # w2 should have been updated too.
    asserts(w, n, w2, n2)

    n3, w3 = random_arrays(NUM_ELS, 'float32')
    w = np.add(w, w3, out=w)
    n = np.add(n, n3, out=n)

    asserts(w, n, w2, n2)
    assert np.allclose(w3, n3)

    # check scalars
    w += 5.0
    n += 5.0
    w.evaluate()

    asserts(w, n, w2, n2)

def test_views_update_mix():
    '''
    '''
    n, w = random_arrays(10, 'float32')
    # Let's add more complexity. Before messing with child views etc, first
    # register an op with the parent as well.
    n = np.sqrt(n)
    w = np.sqrt(w)
    # get the child views
    w2 = w[2:5]
    n2 = n[2:5]

    # updatig the values in place is still reflected correctly.
    w = np.log(w, out=w)
    n = np.log(n, out=n)

    # evaluating this causes the internal representation to change. So can't
    # rely on w.weldobj.context[w.name] anymore.
    w.evaluate()

    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    w2.evaluate()

    assert np.allclose(w[2:5], w2)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w, n)

def test_views_mix2():
    '''
    update parent/child, binary/unary ops.
    '''
    NUM_ELS = 10
    n, w = random_arrays(NUM_ELS, 'float32')

    w2 = w[2:5]
    n2 = n[2:5]

    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    w2.evaluate()

    assert np.allclose(w[2:5], w2)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w, n)

    n3, w3 = random_arrays(NUM_ELS, 'float32')
    w = np.add(w, w3, out=w)
    n = np.add(n, n3, out=n)

    assert np.allclose(w[2:5].evaluate(), w2.evaluate())
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w, n)

    # now update the child

def test_views_grandparents_update_mix():
    '''
    Similar to above. Ensure consistency of views of views etc.
    '''
    n, w = random_arrays(10, 'float32')
    # Let's add more complexity. Before messing with child views etc, first
    # register an op with the parent as well.

    # TODO: uncomment.
    n = np.sqrt(n)
    w = np.sqrt(w)

    # get the child views
    w2 = w[2:9]
    n2 = n[2:9]

    print(w2._weldarray_view.start)
    print(w2._weldarray_view.end)

    w3 = w2[2:4]
    n3 = n2[2:4]

    print(w3._weldarray_view.start)
    print(w3._weldarray_view.end)

    assert np.allclose(w3.evaluate(), n3)

    # updatig the values in place is still reflected correctly.
    w = np.log(w, out=w)
    n = np.log(n, out=n)

    # evaluating this causes the internal representation to change. So can't
    # rely on w.weldobj.context[w.name] anymore.
    w.evaluate()

    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    # w2.evaluate()

    w3 = np.sqrt(w3, out=w3)
    n3 = np.sqrt(n3, out=n3)

    assert np.allclose(w[2:9], w2)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w3.evaluate(), n3)
    assert np.allclose(w, n)
    assert np.allclose(w2[2:4], w3)

def test_views_check_old():
    '''
    Old views should still be valid etc.
    '''
    pass

def test_views_mess():
    '''
    More complicated versions of the views test.
    '''
    # parent arrays
    NUM_ELS = 100
    num_views = 10

    n, w = random_arrays(NUM_ELS, 'float32')
    # in order to avoid sqrt running into bad values
    w += 1000.00
    n += 1000.00

    weld_views = []
    np_views = []

    weld_views2 = []
    np_views2 = []

    for i in range(num_views):
        nums = random.sample(range(0,NUM_ELS), 2)
        start = min(nums)
        end = max(nums)
        # FIXME: Need to add correct behaviour in this case.
        if start == end:
            continue

        weld_views.append(w[start:end])
        np_views.append(n[start:end])

        np.sqrt(weld_views[i], out=weld_views[i])
        np.sqrt(np_views[i], out=np_views[i])

        np.log(weld_views[i], out=weld_views[i])
        np.log(np_views[i], out=np_views[i])

        np.exp(weld_views[i], out=weld_views[i])
        np.exp(np_views[i], out=np_views[i])

        # add some binary ops.
        n2, w2 = random_arrays(len(np_views[i]), 'float32')
        weld_views[i] = np.add(weld_views[i], w2, out=weld_views[i])
        np_views[i] = np.add(np_views[i], n2, out=np_views[i])
        weld_views[i].evaluate()

        a = np.log(weld_views[i].evaluate())
        b = np.log(np_views[i])
        assert np.allclose(a.evaluate(), b)

    w = np.sqrt(w, out=w)
    n = np.sqrt(n, out=n)

    assert np.allclose(n, w)
    # assert np.array_equal(w.evaluate(), n)
    for i, _ in enumerate(w):
        assert w[i] == n[i]

    # assert np.array_equal(n, w.evaluate())

    # TODO: Add stuff with grandchildren, and so on.
    for i in range(num_views):
        assert np.array_equal(np_views[i], weld_views[i].evaluate())
        assert np.allclose(np_views[i], weld_views[i].evaluate())

def test_views_overlap():
    '''
    Two overlapping views of the same array. Updating one must result in the
    other being updated too.
    '''
    NUM_ELS = 10
    dtype = 'float64'
    n, w = random_arrays(NUM_ELS, dtype)

    w2 = w[2:5]
    n2 = n[2:5]

    # TODO: uncomment
    w3 = w[4:7]
    n3 = n[4:7]

    # w4, n4 are non overlapping views. Values should never change
    w4 = w[7:9]
    n4 = n[7:9]

    # w5, n5 are contained within w2, n2.
    w5 = w[3:4]
    n5 = n[3:4]

    # unary part
    w2 = np.exp(w2, out=w2)
    n2 = np.exp(n2, out=n2)
    w2.evaluate()

    assert np.allclose(w[2:5], w2)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w, n)
    assert np.allclose(w5.evaluate(), n5)
    # TODO: check why w4 did not need evaluate.
    assert np.allclose(w4, n4)
    assert np.allclose(w3.evaluate(), n3)

    # binary part:
    # now update the child with binary op
    n3, w3 = random_arrays(3, dtype)
    # n3, w3 = given_arrays([1.0, 1.0, 1.0], 'float32')

    n2 = np.add(n2, n3, out=n2)
    w2 = np.add(w2, w3, out=w2)

    # assert np.allclose(w[2:5], w2)
    assert np.allclose(w, n)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w5.evaluate(), n5)
    assert np.allclose(w4, n4)
    assert np.allclose(w3, n3)

    w2 += 5.0
    n2 += 5.0
    w2.evaluate()

    assert np.allclose(w[2:5], w2)
    assert np.allclose(w, n)
    assert np.allclose(w2.evaluate(), n2)
    assert np.allclose(w5.evaluate(), n5)
    assert np.allclose(w4, n4)
    assert np.allclose(w3, n3)

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
    Special case of broadcasting rules - the scalar is applied to all the
    Weldrray members.
    '''
    t = "int32"
    n, w = random_arrays(NUM_ELS, t)
    n2 = n + 2
    w2 = w + 2

    w2 = w2.evaluate()
    assert np.allclose(w2, n2)

    # test by combining it with binary op.
    n, w = random_arrays(NUM_ELS, t)
    w += 10
    n += 10

    n2, w2 = random_arrays(NUM_ELS, t)

    w = np.add(w, w2)
    n = np.add(n, n2)

    assert np.allclose(w, n)

    t = "float32"
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
    assert id(w) != id(w2)

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
    w3 = w3.evaluate()
    w3 = w3.evaluate()

    assert np.allclose(w3, n3)

def test_setitem_basic():
    '''
    set an arbitrary item in the array after registering ops on it.
    '''
    # TODO: run this on all types.
    n, w = random_arrays(NUM_ELS, 'float32')
    n[0] = 5.0
    w[0] = 5.0
    assert np.allclose(n, w)

    n[0] += 10.0
    w[0] += 10.0
    assert np.allclose(n, w)

    n[2] -= 5.0
    w[2] -= 5.0

    assert np.allclose(n, w)

def test_setitem_slice():
    '''
    '''
    n, w = random_arrays(NUM_ELS, 'float32')

    n[0:2] += 1.0
    w[0:2] += 1.0
    n[0:2] = [5.0, 2.0]
    w[0:2] = [5.0, 2.0]
    assert np.allclose(n, w)

    n[4:6] += 10.0
    # w2 = w[4:6]
    w[4:6] += 10.0
    w = w.evaluate()
    print('n: ', n)
    print('w: ', w)

    assert np.allclose(n, w)

def test_setitem_strides():
    '''
    TODO: make more complicated versions which do multiple types of changes on strides at once.
    TODO2: need to support different strides.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')

    n[0:2:1] = [5.0, 2.0]
    w[0:2:1] = [5.0, 2.0]
    assert np.allclose(n, w)

    n[5:8:1] += 10.0
    w[5:8:1] += 10.0
    assert np.allclose(n, w)

def test_setitem_list():
    '''
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    a = [0, 3]
    n[a] = [5.0, 13.0]
    w[a] = [5.0, 13.0]
    assert np.allclose(n, w)

def test_setitem_weird_indexing():
    '''
    try to confuse the weldarray with different indexing patterns.
    '''
    pass

def test_setitem_mix():
    '''
    Mix all setitem stuff / and other ops.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n = np.sqrt(n)
    w = np.sqrt(w)
    # assert np.allclose(n, w)

    n, w = random_arrays(NUM_ELS, 'float32')

    n[0:2] = [5.0, 2.0]
    w[0:2] = [5.0, 2.0]
    assert np.allclose(n, w)

    n[4:6] += 10.0
    w[4:6] += 10.0
    assert np.allclose(n, w)

def test_setitem_views():
    '''
    What if you use setitem on a view? Will the changes be correctly propagated to the base array
    etc?
    '''
    NUM_ELS = 6
    n, w = random_arrays(NUM_ELS, 'float32')

    n2 = n[1:5]
    w2 = w[1:5]

    # first add stuff to be evaluated
    n += 2.0
    w += 2.0

    n2[0:2:1] = [5.0, 2.0]
    w2[0:2:1] = [5.0, 2.0]

    assert np.allclose(n, w.evaluate())
    assert np.allclose(n2, w2.evaluate())

    print('going to call += ')
    print('going to call += ')
    print('going to call += ')
    n2[0:3:1] += 10.0
    w2[0:3:1] += 10.0

    print('n2: ', n2)
    print('w2: ', w2)

    w = w.evaluate()
    w2 = w2.evaluate()
    assert np.allclose(n2, w2)
    assert np.allclose(n, w)

def test_iterator():
    n, w = random_arrays(NUM_ELS, 'float32')

    w = np.exp(w, out=w)
    n = np.exp(n, out=n)

    for i, e in enumerate(w):
        assert e == n[i]
        assert w[i] == n[i]

def test_views_double_update():
    '''
    Cool edge case involving views / and ordering of np.add args etc.  When using wv = np.add(a,
    b, out=b), other is b, and result is b too. So b gets added to b instead of a.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')

    w += 100.00
    n += 100.00

    wv = w[3:5]
    nv = n[3:5]

    nv2, wv2 = random_arrays(len(wv), 'float32')

    wv = np.add(wv2, wv, out=wv)
    nv = np.add(nv2, nv, out=nv)

    # Instead, this would work:
    # wv = np.add(wv, wv2, out=wv)
    # nv = np.add(nv, nv2, out=nv)

    assert np.allclose(w, n)
    assert np.allclose(wv.evaluate(), nv)

def test_views_strides():
    '''
    Generating views with different strides besides 1.
    FIXME: not supported yet.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    print("creating the view")
    w2 = w[2:8:2]
    n2 = n[2:8:2]

    w += 100.00
    n += 100.00

    w = w.evaluate()
    assert np.allclose(w, n)
    assert np.allclose(w2, n2)

    n2 = np.sqrt(n2, out=n2)
    w2 = np.sqrt(w2, out=w2)

    w = w.evaluate()
    w2 = w2.evaluate()

    assert np.allclose(w2, n2)
    assert np.allclose(w, n)

def test_views_other_indexing():
    '''
    Testing more unusual indexing patterns here.
    This should be much more relevant in multidimensional arrays, so not testing it in depth here.
    '''
    def test_stuff(w, n, w2, n2):

        w += 100.00
        n += 100.00

        assert np.allclose(w, n)
        assert np.allclose(w2.evaluate(), n2)

        w2 = np.sqrt(w2, out=w2)
        n2 = np.sqrt(n2, out=n2)

        assert np.allclose(w, n)
        assert np.allclose(w2.evaluate(), n2)

    n, w = random_arrays(NUM_ELS, 'float32')
    w2 = w[:]
    n2 = n[:]
    test_stuff(w, n, w2, n2)

    w3 = w[2:]
    n3 = n[2:]
    test_stuff(w, n, w2, n2)

# Bunch of failing / error handling tests.
def test_unsupported_views_empty_index():
    n, w = random_arrays(NUM_ELS, 'float32')
    w2 = w[2:2]
    n2 = n[2:2]
    # Fails on this one - but instead this case should be dealt with correctly when setting up
    # inputs.
    assert np.allclose(w2, n2)

def test_unsupported_nan_vals():
    '''
    need to send this off to np to handle as weld fails if elements are nans etc.
    '''
    n, w = random_arrays(100, 'float32')
    for i in range(2):
        n = np.exp(n)
        w = np.exp(w)
    assert np.allclose(n, w)

def test_unsupported_types():
    n, w = given_arrays([2.0, 3.0], 'float32')
    t = np.array([True, False])
    n = n*t
    w = w*t
    assert np.allclose(n, w)

    n, w = given_arrays([2.0, 3.0], 'float32')

    # Not sure what input this is in ufunc terms
    n = n*True
    w = w*True
    assert np.allclose(n, w)

def test_unsupported_ndarray_output():
    '''
    kind of a stupid test - just make sure weldarray doesn't die with ugly errors.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')
    n = np.exp(n, out=n)
    n2 = np.exp(w, out=n2)
    assert np.allclose(n,n2)

def test_new_array_creation():
    '''
    Creating new array with an op should leave the value in the old array unchanged.
    If the weldobject.evaluate() method would perform the update in place, then this test would
    fail.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2 = np.sqrt(n)
    w2 = np.sqrt(w)

    assert np.allclose(n, w)
    assert np.allclose(n2, w2)

def test_reduce():
    '''
    reductions is another type of ufunc. Only applies to binary ops. Not many other interesting
    cases to test this because it just evaluates stuff and returns an int/float.
    '''
    for t in TYPES:
        for r in REDUCE_UFUNCS:
            n, w = random_arrays(NUM_ELS, t)
            n2 = r(n)
            w2 = r(w)
            assert np.allclose(n2, w2)

def test_vectorization_bug():
    '''
    simplest case of a bug that seems to occur in more complicated programs with different order of
    arrays etc. Seems to happen because of the vectorization pass.
    '''
    # minimum case to reproduce bug:
    n, w = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')

    n = n*2.0
    w = w*2.0

    n2 = n + n2
    w2 = w + w2

    # Note: Here y + x produces the correct result! (weld IR given below) or if we evaluate y before.
    n3 = n + n2
    w3 = w + w2
    # this produces correct result:
    # w3 = w2 + w

    w3 = w3.evaluate()

    assert np.allclose(n, w)
    assert np.allclose(n2, w2)
    assert np.allclose(n3, w3)

def test_blackscholes_bug():
    '''
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2, w2 = random_arrays(NUM_ELS, 'float32')
    n3, w3 = random_arrays(NUM_ELS, 'float32')

    n4 = n - (np.exp(n2) * n3)
    w4 = w - (np.exp(w2) * w3)

    assert np.allclose(n4, w4)

# FIXME: erf can be routed through array ufuncs.
def test_erf():
    '''
    Separate test because numpy and weld have different functions for this right now.
    '''
    for dtype in TYPES:
        # int still not supported for the unary ops in Weld.
        if "int" in dtype:
            continue
        n, w = random_arrays(NUM_ELS, dtype)

        n2 = ss.erf(n)
        w2 = ss.erf(w)

        w2_eval = w2.evaluate()

        assert np.allclose(w2, n2)

        # TODO: this works with all other unary ops but doesn't work with erf...need to debug it
        # further. Might have something to do with the fact that erf is not routed through
        # __array_ufunc__.
        # assert np.array_equal(w2_eval, n2)

def test_views_inplace_evaluate():
    '''
    If you update a view in place and then evaluate it -- the returned weldarray should still have
    a _weldarray_view.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n2 = n[2:4]
    w2 = w[2:4]
    w2 += 5.0
    assert w2._weldarray_view is not None
    w2 = w2.evaluate()
    # Originally, we were not preserving the _weldarray_view after evaluate.
    assert w2._weldarray_view is not None

def test_array_equal():
    '''
    np.array_equal DOES Not pass subclasses along while evaluating -- and on our side, this
    fails because of a subtle reason somehow...
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    # in order to avoid sqrt running into bad values
    w += 1000.00
    n += 1000.00
    w = np.log(w)
    n = np.log(n)

    # allclose OR w.evaluate() first, and then array_equal leads to mysterious failing (unless
    # we return a new weldarray from evaluate ...)
    print(w.weldobj.context)
    w.evaluate()
    assert np.allclose(n, w)
    assert np.array_equal(n,w.evaluate())

def test_setitem_float():
    '''
    can't automatically assume value to be added is a list as we were doing.
    '''
    n, w = random_arrays(NUM_ELS, 'float32')
    n[0:5:1] = 2.0
    w[0:5:1] = 2.0
    assert np.allclose(n, w)

def test_setitem_list():
    '''
    idx is a list.
    '''
    n, w = random_arrays(10, 'float32')
    n[[2,4]] = 4.0
    w[[2,4]] = 4.0
    assert np.allclose(n, w)
    n[2] = 2.0
    w[2] = 2.0
    assert np.allclose(n, w)

def test_too_many_ops():
    '''
    causes segault because there are too many ops for the weld graph to handle.
    '''
    n, w = random_arrays(10, 'float32')
    for i in range(1000):
        w += 10.0
        n += 10.0
    w = w.evaluate()
    assert np.allclose(w, n)

def test_non_matching_types():
    for t in TYPES:
        n, w = random_arrays(10, t)
        if ('int' in t):
            n += 10
            w += 10
        else:
            n += 10.0
            w += 10.0

    np.allclose(n, w)

def test_let_sorting():
    '''
    FIXME: test doesn't quite test what is needed.
    Want to generate a situation like this, from TestArmedTestBeds.py in rl
    algorithms repo:
        |_inp0: vec[f64], _inp2007: vec[f64], _inp2009: vec[f64]| let obj10001 = (_inp2009);
        let obj10002 = (_inp0);
        let obj10003 = (result(for(zip(obj10002,obj10001), appender,|b,i,e| merge(b,e.$0 - e.$1))));
        let obj10004 = (result(for(obj10003, appender,|b,i,e| merge(b,-0.197762439964 *e))));
        let obj10005 = (result(for(zip(obj9996,obj10004), appender,|b,i,e| merge(b,e.$0 + e.$1))));
        let obj9900 = (_inp0);
        let obj9992 = (_inp2007);
        let obj9993 = (_inp0);
        let obj9994 = (result(for(zip(obj9993,obj9992), appender,|b,i,e| merge(b,e.$0 - e.$1))));
        let obj9995 = (result(for(obj9994, appender,|b,i,e| merge(b,0.0 *e))));
        let obj9996 = (result(for(zip(obj9900,obj9995), appender,|b,i,e| merge(b,e.$0 + e.$1))));
        result(for(obj10005, appender,|b,i,e| merge(b,exp(e)))): Undefined symbol obj9996
    '''
    weld_arrays = []
    for i in range(1020):
        # n, w = aiven_arrays([1.0, 1.0, 1.0], 'float32')
        n, w = random_arrays(5, 'float32')
        weld_arrays.append(w)
        if i == 0:
            weld_arrays[0] = np.sqrt(weld_arrays[0])
        if i == 1:
            weld_arrays[0] = weld_arrays[0] + weld_arrays[1]

    weld_arrays[1010] = np.sqrt(weld_arrays[1010])
    weld_arrays[0] = np.sqrt(weld_arrays[0])
    for i in range(1020):
        if i == 0:
            weld_arrays[0] = np.sqrt(weld_arrays[0])
        if i == 1:
            weld_arrays[0] = weld_arrays[0] + weld_arrays[1]


    w2 = weld_arrays[0] + weld_arrays[1010]
    print(w2.weldobj.weld_code)
    w2 = w2.evaluate()

def test_util_functions():
    '''
    looks stuff like np.max etc. work fine without needing to evaluate -->
    evaluate is being called implicitly in np.amax's implementation.
    '''
    for f in UTIL_FUNCTIONS:
        n, w = random_arrays(5, 'float64')
        n = np.log(n)
        w = np.log(w)
        nmax = f(n)
        wmax = f(w)
        assert nmax == wmax
