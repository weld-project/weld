import weldnumpy as wn
import numpy as np

def assert_correct(np_a, z):
    '''
    common part of the check.
    '''
    shape = []
    for s in z.shape:
        shape.append(s)
    shape = tuple(shape)

    np_a = np.reshape(np_a, shape)

    for i in range((z.shape[0])):
        for j in range(z.shape[1]):
            assert np_a[i][j] == z[i][j]

def test_view():
    '''
    Adding the iter code to a view.
    In general, do we deal correctly with wn.array(view)??
    '''
    orig = np.random.rand(20,20)
    a = orig[3:15:1,4:20:2]

    # orig = np.random.rand(20,20,20)
    # a = orig[3:15:3,:,:]

    print(a.flags)
    assert not a.flags.contiguous
    a = wn.array(a)
    z = np.copy(a)

    shapes = a.weldobj.update(np.array(list(a.shape)))

    strides = []
    for s in a.strides:
        strides.append(s/z.itemsize)

    strides = a.weldobj.update(np.array(strides))

    end = 1
    for s in a.shape:
        end = end*s
    end = end

    iter_code = 'result(for(nditer({arr}, 0L, {end}L, 1L, {shapes}, {strides}), appender, \
    |b, i, e| merge(b,exp(e))))'.format(shapes=shapes, strides=strides, end=str(end), arr=a.name)

    a.weldobj.weld_code = iter_code
    z = np.exp(z)
    # convert the data represented by weldarray 'a', to a multi-dimensional numpy array of shape as z,
    # and then compare the values.
    np_a = a._eval()
    assert_correct(np_a, z)


def test_start():
    '''
    Has a different start from the base array.
    '''
    orig = np.random.rand(20,20)
    a = orig[0:20:1,0:20:1]

    start = (wn.addr(a) - wn.addr(orig)) / a.itemsize
    orig = wn.array(orig)
    z = np.copy(a)
    z = np.exp(z)

    shapes = orig.weldobj.update(np.array(list(a.shape)))

    strides = []
    for s in a.strides:
        strides.append(s/8)

    strides = orig.weldobj.update(np.array(strides))

    end = 1
    for s in a.shape:
        end = end*s
    end = end + start

    iter_code = 'result(for(nditer({arr}, {start}L, {end}L, 1L, {shapes}, {strides}), appender, \
    |b, i, e| merge(b,exp(e))))'.format(shapes=shapes, strides=strides, end=str(end),
            start=str(start), arr=orig.name)
    orig.weldobj.weld_code = iter_code
    np_a = orig._eval()

    assert_correct(np_a, z)

def test_zip():
    '''
    Has a different start from the base array.
    '''
    orig = np.random.rand(20,20)
    orig2 = np.random.rand(20,20)
    a = orig[5:20:1,3:20:2]
    b = orig2[5:20:1,3:20:2]
    start = (wn.addr(a) - wn.addr(orig)) / a.itemsize
    orig = wn.array(orig)

    # copying so we can test them later.
    z = np.copy(a)
    z2 = np.copy(b)
    # added orig2 to orig's weldobject.
    orig_2_name = orig.weldobj.update(orig2)


    shapes = orig.weldobj.update(np.array(list(a.shape)))

    strides = []
    for s in a.strides:
        strides.append(s/8)

    strides = orig.weldobj.update(np.array(strides))

    end = 1
    for s in a.shape:
        end = end*s
    end = end + start

    iter_code = 'result(for(zip(nditer({arr}, {start}l, {end}l, 1l, {shapes}, {strides}),  \
nditer({arr2}, {start}l, {end}l, 1l, {shapes}, {strides})), \
appender, |b, i, e| merge(b,e.$0+e.$1)))'.format(shapes=shapes, strides=strides, end=str(end),
start=str(start), arr=orig.name, arr2=orig_2_name)
    orig.weldobj.weld_code = iter_code
    # gives us a numpy array after evaluating the nditer code above.
    np_a = orig._eval()
    # update the copied array.
    z3 = z+z2;
    # test values are equal.
    assert_correct(np_a, z3)

# few different tests.
test_view()
test_start()
test_zip()
