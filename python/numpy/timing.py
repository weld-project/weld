import numpy as np
import weldarray as wa
import py.test
import random
import timeit
import time

def simple_loop_binary(lib, reps):
    '''
    lib = np or wa.
    '''
    test = []
    test2 = []
    test3 = []
    for i in range(100):
        test.append(float(random.random()))
        test2.append(float(random.random()))
        test3.append(float(random.random()))

    arr = lib.array(test, dtype="float32")
    arr2 = lib.array(test2, dtype="float32")
    arr3 = lib.array(test3, dtype="float32")
    big_array = []
    for i in range(reps):

        # FIXME: For in place ops, weld is 5-10 times slower than when the left
        # hand side is a different array...
        # arr = arr + arr2

        arr3 = arr - arr3
        arr3 = arr*arr2

    print(arr)

def simple_loop_fusion():
    pass

# timeit.timeit('simple_loop_binary', wa, 10000, 2)
start = time.time()
simple_loop_binary(np, 10000)
print("loop took {} seconds".format(time.time() - start))
