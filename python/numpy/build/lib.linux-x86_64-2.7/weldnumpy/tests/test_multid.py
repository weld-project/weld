import numpy as np
import py.test
import random
from weldnumpy import *
import weldnumpy as wn
from test_utils import *

'''
Need to add a bunch of _real_shape invariance based tests.
'''

def diagonal(ary, offset=0):
    """
    from bohrium guys.
    """
    if ary.ndim != 2:
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (ary.shape[0]-offset) > ary.shape[1]:
            ary_diag = ary[offset, :]
        else:
            ary_diag = ary[offset:, 0]
    else:
        if ary.shape[1]-offset > ary.shape[0]:
            ary_diag = ary[:, offset]
        else:
            ary_diag = ary[0, offset:]
    ary_diag.strides = (ary.strides[0]+ary.strides[1],)
    if isinstance(ary_diag, weldarray):
        ary_diag._weldarray_view.strides = ary_diag.strides

    return ary_diag


'''
General Notes:
    - Two multi-dim contig arrays with different strides/shapes isn't possible.
'''

'''
Views based tests.

TODO tests:
    - Scalars with multi-dim arrays.
    - somehow test that for non-contig arrays, weld's loop goes over as much contiguous elements as
      it can.
    - scalar op view.
    - a + other (view).
    - Then update the view after. This should not affect a+other's result.
'''
# ND_SHAPES = [(5,3,4), (5,4), (6,4,7,3)]
ND_SHAPES = [(5,3,4), (3,4)]

def get_noncontig_idx(shape):
    '''
    Returns indices that make an array with shape, 'shape', become a non contiguous array.

    Just gives a bunch of random multi-d indices.
    Things to test:
        - different number of idxs
        - different types of indexing styles (None, :, ... etc.)
        - different sizes, strides etc.
    '''
    # idx = []
    
    # 1000 tries to find a non-contig idx. 5 was actually failing to find a
    # non-contig case sometimes...once the correct idx is found, will break out
    # of loop, so should be much shorter than 1000.
    for i in range(1000):
        idx = []
        for s in shape:
            # 5 tries to get non-contiguous array
            start = random.randint(0, s-1)
            stop = random.randint(start+1, s)
            # step = random.randint(0, 3)
            step = 1
            idx.append(slice(start, stop, step))

        idx = tuple(idx)
        a, _ = random_arrays(shape, 'float32')
        b = a[idx]
        if is_view_child(b,a) and not b.flags.contiguous:
            break
    
    return idx

def test_views_non_contig_basic():
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float64')
        idx = get_noncontig_idx(shape)
        n2 = n[idx]
        w2 = w[idx]

        # useful test to add.
        assert w2.shape == n2.shape
        assert w2.flags == n2.flags
        assert w2.strides == n2.strides
        # test unary op.
        n3 = np.sqrt(n2)
        w3 = np.sqrt(w2)
        w3 = w3.evaluate()

        assert np.array_equal(n, w)
        assert np.array_equal(n2, w2)
        assert np.array_equal(n3, w3)

        # allclose seems harder to get to work!
        assert np.allclose(n, w)
        assert np.allclose(n2, w2)
        assert np.allclose(n3, w3)

    # test binary op.

def test_views_non_contig_no_op():
    '''
    This just requires the sub/mult etc. used for allclose to work.
    '''
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)

        n2 = n[idx]
        w2 = w[idx]

        assert w2._weldarray_view is not None
        w2 = w2.evaluate()
        assert w2._weldarray_view is not None

        assert n2.shape == w2.shape
        assert n2.strides == w2.strides

        assert np.allclose(n, w)
        assert np.array_equal(n2, w2)
        print("************array equal done************")
        assert np.allclose(n2, w2)
        assert np.allclose(n2, w2.evaluate())

def test_views_non_contig_inplace_unary():

    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float64')
        idx = get_noncontig_idx(shape)
        n2 = n[idx]
        w2 = w[idx] 

        assert is_view_child(n2, n), 'should be child'
        assert is_view_child(w2.view(np.ndarray), w.view(np.ndarray)), 'should be child'
 
        # test: update parents first
        n = np.log(n, out=n)
        w = np.log(w, out=w)

        assert np.allclose(n2, w2)
        assert np.allclose(n, w)

        # unary op test.
        n2 = np.sqrt(n2, out=n2)
        w2 = np.sqrt(w2, out=w2)

        assert n2.shape == w2.shape
        assert n2.strides == w2.strides

        print('w: ', w)
        print('n: ', n)

        print('w2: ', w2)
        print('n2: ', n2)

        w2 = w2.evaluate()
        assert np.array_equal(n2, w2)
        assert np.allclose(n2, w2.evaluate())
        assert np.allclose(n, w.evaluate())

def test_views_non_contig_newarray_binary():
    '''
    binary involves a few cases that we need to test:
        - non-contig + contig
        - contig + non-contig
        - non-contig + non-contig
    '''
    ND_SHAPES = [(3,3)]
    BINARY_OPS = [np.add]

    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float64')
        n2, w2 = random_arrays(shape, 'float64')
        idx = (slice(0,3,1), slice(2,3,1))

        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]

        for op in BINARY_OPS:
            nv3 = op(nv1, nv2)
            wv3 = op(wv1, wv2)
            wv3 = wv3.evaluate()

            assert nv3.shape == wv3.shape, 'shape not same'
            assert np.allclose(nv3, wv3)
            # FIXME: Need to try and make it so we don't have to call evaluate when using
            # array_equal, or when its on non-views.
            assert np.allclose(nv1, wv1.evaluate())
            assert np.allclose(nv2, wv2.evaluate())
            assert np.array_equal(nv1, wv1)
            assert np.array_equal(nv2, wv2)

def test_views_non_contig_inplace_binary1():
    '''
    FIXME: Heisenbug
    TODO: separate out case with updated other into new test.

    binary involves a few cases that we need to test:
    In place ops, consider the case:
        - non-contig + non-contig

    Note: these don't test for performance, for instance, if the non-contig array is being
    evaluated with the max contiguity etc.
    '''
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        n2, w2 = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)
        print('idx : ', idx)

        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]

        # update other from before.
        # important: we need to make sure the case with the second array having some operations
        # stored in it is dealt with.
        wv2 = np.sqrt(wv2, out=wv2)
        nv2 = np.sqrt(nv2, out=nv2)

        for op in BINARY_OPS:
            print('op = ', op)
            nv1 = op(nv1, nv2, out=nv1)
            wv1 = op(wv1, wv2, out=wv1)

            # when we evaluate a weldarray_view, the view properties (parent array etc) must be preserved in the
            # returned array.
            wv1 = wv1.evaluate()

            assert np.allclose(nv2, wv2.evaluate())
            assert np.allclose(nv1, wv1)
            assert np.allclose(n, w)
            assert np.allclose(n2, w2)

            print('***********ENDED op {} ************'.format(op))
        print('***********ENDED shape {} ************'.format(shape))

# def test_views_non_contig_inplace_binary2():
    # '''
    # binary involves a few cases that we need to test:
    # In place ops:
        # - contig + non-contig
        # - non-contig + contig
            # - surprisingly these two cases aren't so trivial because we can't just loop over the
              # contiguous array as there would be no clear way to map the indices to the non-contig
              # case.
            # - it might even be better sometimes for performance to convert the non-contig case to
              # contig?

    # These cases are similar to the newarray cases though and both these tests should pass together
    # based on current implementation.
    # TODO: write the test.
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)
        # # TODO: write test.
        # pass

def test_views_non_contig_nested_child():
    '''
    non-contig view getting another view made off it. Second view seems to fail
    getting a _weldarray view.
    '''
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)

        nv1 = n[idx]
        wv1 = w[idx]

        assert wv1._weldarray_view is not None
        
        # FIXME: make this more general somehow + this should actually be
        # asserting in weldnumpy?
        idx2 = slice(0,-1,2)
        nv2 = nv1[idx2]
        wv2 = wv1[idx2]

        assert wv2._weldarray_view is not None

def test_weldarray_vanishing_view():
    '''
    mysteriously vanishing weldarray view...
    '''
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)

        nv = n[idx]
        wv = w[idx]
        assert wv._weldarray_view is not None

        # update other from before.
        # important: we need to make sure the case with the second array having some operations
        # stored in it is dealt with.
        wv = np.sqrt(wv, out=wv)
        nv = np.sqrt(nv, out=nv) 
        assert wv._weldarray_view is not None

        wv = wv.evaluate()

        assert wv._weldarray_view is not None

def test_views_non_contig_inplace_other_updates():
    '''
    FIXME: This fails because non-contig in place updates.
        - if non-contiguous array has in place update, then we offload to numpy --- but at least
          right now, there is no way to keep the _weldarray_view of the non-contiguous view intact!
        - potentially some other problems here too!
    '''
    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        n2, w2 = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)

        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]
        assert wv2._weldarray_view is not None

        # update other from before.
        # important: we need to make sure the case with the second array having some operations
        # stored in it is dealt with.
        wv2 = np.sqrt(wv2, out=wv2)
        nv2 = np.sqrt(nv2, out=nv2)
        wv2 = wv2.evaluate()

        assert wv2._weldarray_view is not None

        op = np.subtract
        nv1 = op(nv1, nv2, out=nv1)
        wv1 = op(wv1, wv2, out=wv1)

        nv2 = np.log(nv2, out=nv2)
        wv2 = np.log(wv2, out=wv2)
        wv2 = wv2.evaluate()

        n2 = np.sqrt(n2, out=n2)
        w2 = np.sqrt(w2, out=w2)

        # when we evaluate a weldarray_view, the view properties (parent array etc) must be preserved in the
        # returned array.
        wv1 = wv1.evaluate()
        wv2 = wv2.evaluate()
        assert wv2._weldarray_view is not None

        assert np.array_equal(nv2, wv2)
        assert np.allclose(nv2, wv2)
        assert np.allclose(nv1, wv1)
        assert np.allclose(n, w)
        assert np.allclose(n2, w2)

        print('***********ENDED shape {} ************'.format(shape))


# def test_views_non_contig_inplace_binary_mess():
    # '''
    # Just mixes a bunch of operations on top of the previous setup.
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)
        # nv1 = n[idx]
        # wv1 = w[idx]
        # nv2 = n2[idx]
        # wv2 = w2[idx]
        # # TODO: write test.

'''
General Tests.
'''
def test_reshape():
    n, w = random_arrays(36, 'float32')
    n = reshape(n, (6,6))
    w = reshape(w, (6,6))

    assert isinstance(w, weldarray)
    assert w.shape == n.shape
    assert w.strides == n.strides
    w._real_shape = w.shape

    assert np.allclose(n, w)

    n2, w2 = random_arrays((6,6), 'float32')
    n = reshape(n2, 36)
    w = reshape(w2, 36)
    assert isinstance(w, weldarray)
    assert w.shape == n.shape
    assert w.strides == n.strides
    assert np.allclose(n, w)

'''
TODO: Set up multi-dimensional array creation routines.

    - create new arrays in different ways: transpose, reshape, concatenation, horizontal/vertical etc.
'''
def test_subtle_new_array():
    pass
    '''
    FIXME:
    a.T, or a.imag seem to create new weldarrays without passing through __init__. Might have to
    create __array_finalize__ after all.
    '''
    # for a in attr:
        # if a == 'imag' or a == 'T':
            # continue
        # print(a)
        # print(eval('n.' + a))
        # print(eval('w.' + a))

'''
More advanced tests.
'''

'''
Array transformations based tests.
'''
def test_transpose_simple():
    n, w = random_arrays((10,10), 'float64')
    # slightly awkward because otherwise numpy's transpose will be called...ideally, weldnumpy is
    # being used as import weldnumpy as np, and then this stuff would work fine.
    n2 = transpose(n)
    w2 = transpose(w)

    assert n2.shape == w2.shape
    assert w2.strides == n2.strides
    # import test
    assert w2._weldarray_view is not None, 'should not be None!'

# FIXME: Annoying failure in the weld/rust code.
# def test_transpose_ops():
    # '''
    # FAILS WITH A unwrap on None value error.
    # TODO: do a bunch of things after the transpose. _get_result and views would have to deal
    # correctly without a specific index...
    # This kinds works but the support is pretty flimsy...
    # '''
    # n, w = random_arrays((10,5), 'float64')
    # # slightly awkward because otherwise numpy's transpose will be called...ideally, weldnumpy is
    # # being used as import weldnumpy as np, and then this stuff would work fine.
    # n2 = np.transpose(n)
    # w2 = np.transpose(w)

    # assert n2.shape == w2.shape

    # np.allclose(n2, w2)
    # n3 = np.sqrt(n2)
    # w3 = np.sqrt(w2)
    # w3 = w3.evaluate()

    # assert n3.shape == w3.shape
    # assert w3.flags.contiguous

    # np.allclose(n3, w3)
    # np.allclose(n2, w2)
    # np.allclose(n, w)

    # print("first bunch of allcloses done")

    # n4 = n3 - n2
    # w4 = w3 - w2
    # # THIS WORKS: 
    # # w4 = w2 - w3
    
    # print(w4.weldobj.weld_code)
    # w4 = w4.evaluate()
    # print("w4 evaluate done")

    # np.allclose(n3, w3)
    # np.allclose(n2, w2)
    # np.allclose(n, w)
    # np.allclose(w4, n4)

def test_transpose_inplace():
    '''
    kinda surprised but this actually seems to just work.
    TODO: Need to dig deeper to find edge cases.
    '''
    n, w = random_arrays((10,5), 'float64')
    # slightly awkward because otherwise numpy's transpose will be called...ideally, weldnumpy is
    # being used as import weldnumpy as np, and then this stuff would work fine.
    n2 = n.transpose()
    w2 = np.transpose(w)
    n3 = np.sqrt(n2, out=n2)
    w3 = np.sqrt(w2, out=w2)
    w3 = w3.evaluate()

    np.allclose(n3, w3)
    np.allclose(n2, w2)
    np.allclose(n, w)

    n4, w4 = random_arrays(n2.shape, 'float64')

    n3 -= n4
    w3 -= w4
    w4 = w4.evaluate()
    np.allclose(w4, n4)

def test_broadcasting_simple():
    n, w = random_arrays((1,10), 'float32')
    n2, w2 = random_arrays((10,1), 'float32')

    # broadcasting should apply to these
    n3 = n2 + n
    w3 = w2 + w
    w4 = w + w2
    w3 = w3.evaluate()
    w4 = w4.evaluate()
    assert isinstance(w3, weldarray)
    assert w3.shape == n3.shape == w4.shape
    assert w3.strides == n3.strides == w4.strides
    assert np.allclose(w3.evaluate(), n3)
    assert np.allclose(w4, n3)
    assert np.allclose(w4, w3)

def test_broadcasting_bug():
    '''
    In this case, only one of the arrays is being broadcasted. Some subtle-ish bug seems to show up
    here.
    '''
    n, w = random_arrays((3,2), 'float32')
    n2, w2 = random_arrays(2, 'float32')

    # broadcasting should apply to these
    n3 = n2 + n
    w3 = w + w2
    w4 = w2 + w

    w3 = w3.evaluate()
    w4 = w4.evaluate()

    assert isinstance(w3, weldarray)
    assert w3.shape == n3.shape == w4.shape

    assert w3.strides == n3.strides == w4.strides
    # finally test values too, why not
    assert np.allclose(w3, n3)
    assert np.allclose(w3, w4)
    assert np.allclose(w4, n3)

def test_broadcasting_nbody_bug():
    '''
    Transpose + broadcasting --> shapes don't seem to match.
    '''
    n, w = random_arrays(1000, 'float64')
    a = np.transpose(n[np.newaxis,:])
    print(w._real_shape)
    w2 = w[np.newaxis,:]
    print(type(w2))
    print(w2.shape)
    # print(w2._real_shape)
    b = transpose(w[np.newaxis,:])
    b = b.evaluate()
    print(type(b))
    assert np.allclose(a, b)

    numpy_dx = a - n
    weld_dx = b - w
    weld_dx = weld_dx.evaluate()

    assert numpy_dx.shape == weld_dx.shape, 'shapes must match!'
    assert np.array_equal(numpy_dx, weld_dx)
    assert np.allclose(numpy_dx, weld_dx)

def test_sum_axis0_simple():
    '''
    Just a 2d contiguous array, so the simplest of all cases.
    TODO: need to generalize this for arbitrary dimensions, and for non-contiguous arrays.
    '''
    n, w = random_arrays((7,11), 'float64')
    n2 = np.sum(n, axis=0)
    w2 = np.sum(w, axis=0)
    w2 = w2.evaluate()
    assert np.allclose(w2, n2)

def test_sum_axis1_simple():
    '''
    switching the axis.
    '''
    n, w = random_arrays((18,17), 'float64')
    n2 = np.sum(n, axis=1)
    w2 = np.sum(w, axis=1)
    w2 = w2.evaluate()
    assert np.allclose(w2, n2)

def test_ops_axis():
    pass

def test_diagonal():
    n, w = random_arrays((5,5), 'float64')
    n2 = diagonal(n)
    w2 = diagonal(w)
    # w2 = w2.evaluate()

    assert n2.shape == w2.shape
    assert n2.strides == w2.strides
    assert np.allclose(n2, w2)
    assert np.array_equal(n2, w2)

def test_diagonal_setitem():
    '''
    functionality required for nbody .
    '''
    # wn.set_offload_setitem(True)
    n, w = random_arrays((5,5), 'float64')
    n2 = diagonal(n)
    w2 = diagonal(w)
    n2[:] = 1.0
    w2[:] = 1.0
    # wn.set_offload_setitem(False)
    assert np.allclose(n2, w2)
    assert np.array_equal(n2, w2)
    assert np.array_equal(n, w)

def test_nbody_bug2():
    n, w = random_arrays(100, 'float64')
    a = transpose(n[np.newaxis,:])
    b = transpose(w[np.newaxis,:])

    ndx = a - n
    wdx = b - w

    print(wdx.shape)
    print(wdx._real_shape)

    nr = np.sqrt(ndx**2)
    wr = np.sqrt(wdx**2)
    wr._real_shape = wdx._real_shape

    # wr = wr.evaluate()
    # assert np.allclose(nr, wr)

    nr2 = nr**3
    wr2 = wr**3
    wr2._real_shape = wr._real_shape
    wr2 = wr2.evaluate()

    assert np.allclose(nr2, wr2)

def test_less_than():
    n, w = random_arrays(10, 'float64')

def test_for_loop():
    n, w = random_arrays((5,5), 'float64')
    for i, ni in enumerate(n):
        assert np.allclose(ni,w[i])

def test_for_loop_reshape():
    n, w = random_arrays((5,5), 'float64')
    n2 = n.reshape(5*5)
    for i, wi in enumerate(w.reshape(5*5)):
        np.allclose(wi, n2[i])

def test_tictactoe_reshape():
    '''
    weird behaviour for reshape before, test was taken from tictactoe
    implementation in rl algorithms repo.
    '''
    n, w = random_arrays((5,5), 'float64')
    w2 = w.reshape(5*5)

    # ensures that for loops on reshaped arrays work as expected
    for i in w.reshape(5*5):
        if i == -1:
            print('woot')
        assert isinstance(i, np.float64)

test_views_non_contig_inplace_unary()
