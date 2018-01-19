import numpy as np
import py.test
import random
from weldnumpy import *
from test_utils import *

'''
TODO:
    1. tests that preserve view properties.
'''

'''
Tests that are common for both 1-d and multi-dimensional cases
'''
def test_unary_elemwise():
    '''
    Tests all the unary ops in UNARY_OPS.

    FIXME: For now, unary ops seem to only be supported on floats.
    '''
    for SHAPE in SHAPES:
        for op in UNARY_OPS:
            for dtype in TYPES:
                print(dtype)
                # int still not supported for the unary ops in Weld.
                if "int" in dtype:
                    continue
                np_test, w = random_arrays(SHAPE, dtype)
                w2 = op(w)
                np_result = op(np_test)
                w2_eval = w2.evaluate()

                assert np.allclose(w2, np_result)
                assert np.array_equal(w2_eval, np_result)

def test_binary_elemwise():
    '''
    '''
    for SHAPE in SHAPES:
        for op in BINARY_OPS:
            for dtype in TYPES:
                np_test, w = random_arrays(SHAPE, dtype)
                np_test2, w2 = random_arrays(SHAPE, dtype)
                w3 = op(w, w2)
                weld_result = w3.evaluate()
                np_result = op(np_test, np_test2)
                # Need array equal to keep matching types for weldarray, otherwise
                # allclose tries to subtract floats from ints.
                assert np.array_equal(weld_result, np_result)

def test_mix_np_weld_ops():
    '''
    Weld Ops + Numpy Ops - before executing any of the numpy ops, the
    registered weld ops must be evaluateuated.
    '''
    for SHAPE in SHAPES:
        np_test, w = random_arrays(SHAPE, 'float32')
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
    for SHAPE in SHAPES:
        print('shape = ', SHAPE)
        t = "int32"
        print("t = ", t)
        n, w = random_arrays(SHAPE, t)
        n2 = n + 2
        w2 = w + 2

        w2 = w2.evaluate()
        assert np.allclose(w2, n2)

        # test by combining it with binary op.
        n, w = random_arrays(SHAPE, t)
        w += 10
        n += 10

        n2, w2 = random_arrays(SHAPE, t)

        w = np.add(w, w2)
        n = np.add(n, n2)

        assert np.allclose(w, n)

        t = "float32"
        print("t = ", t)
        np_test, w = random_arrays(SHAPE, t)
        np_result = np_test + 2.00
        w2 = w + 2.00
        weld_result = w2.evaluate()
        assert np.allclose(weld_result, np_result)

def test_shapes():
    '''
    After creating a new array and doing some operations on it - the shape, ndim, and {other
    atrributes?} should remain the same as before.
    '''
    for SHAPE in SHAPES:
        n, w = random_arrays(SHAPE, 'float32')
        n = np.exp(n)
        w = np.exp(w)
        w = w.evaluate()

        assert n.shape == w.shape
        assert n.ndim == w.ndim
        assert n.size == w.size
        assert np.array_equal(w, n)

def test_square():
    '''
    square, and other powers.
    '''
    for s in SHAPES:
        n, w = random_arrays(s, 'float32')
        n2 = np.square(n)
        w2 = np.square(w)
        assert np.allclose(n2, w2)

def test_powers():
    '''
    square, and other powers.
    '''
    POWERS = [3, 5, 8];
    for s in SHAPES:
        for p in POWERS:
            n, w = random_arrays(s, 'float32')
            n2 = np.power(n, p)
            w2 = np.power(w, p)
            assert np.allclose(n2, w2)

def test_less():

    # for s in SHAPES:
        # for op in CMP_OPS:

    s = 10
    op = np.less 
    n, w = random_arrays(s, 'float64')
    print('n = ', n)

    n2 = op(n, 4.0)
    w2 = w._cmp_op(4.0, op.__name__, None)
    w2 = w2.evaluate()
    # casting it as a ndarray because weld does not support arithmetic between booleans / and
    # numbers;
    assert np.allclose(w2.view(np.ndarray), n2)

def test_less_invert():
    s = 10
    cmp_val = 4.0
    n, w = random_arrays(s, 'float64')
    n2 = np.less(n, cmp_val)
    n2 = ~n2
    print(n2)

    w2 = w._cmp_op(cmp_val, np.greater.__name__)
    w2 = w2.evaluate()
    print(w2)

    assert np.allclose(w2.view(np.ndarray), n2)

def test_trig():
    for s in SHAPES:
        for op in TRIG_OPS:
            n, w = random_arrays(s, 'float64')
            n2 = op(n)
            w2 = op(w)

            assert np.allclose(n2, w2)

