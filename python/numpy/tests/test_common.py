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
        print('view: ', w._weldarray_view)
        n = np.exp(n)
        w = np.exp(w)
        print('view: ', w._weldarray_view)
        w = w.evaluate()

        assert n.shape == w.shape
        assert n.ndim == w.ndim
        assert n.size == w.size
        assert np.array_equal(w, n)
