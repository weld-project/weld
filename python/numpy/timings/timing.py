import numpy as np
import sys
sys.path.append('..')
import weldarray as wa
import py.test
import random
import timeit
import time

NUM_REPS = 100000
NUM_ELS = 100000

def simple_loop_binary(lib, reps):
    '''
    lib = np or wa.
    '''
    random.seed(1234)
    test = []
    test2 = []
    test3 = []
    for i in range(NUM_ELS):
        test.append(float(random.random()))
        test2.append(float(random.random()))
        test3.append(float(random.random()))

    arr = lib.array(test, dtype="float32")
    arr2 = lib.array(test2, dtype="float32")
    arr3 = lib.array(test3, dtype="float32")

    for i in range(reps):
        # FIXME: For in place ops, weld is 5-10 times slower than when the left
        # hand side is a different array...
        # arr = arr + arr2
        # arr = arr*arr2
        arr3 = arr + arr2
        arr3 = arr*arr2
    
    print(type(arr3))
    return arr3

def simple_loop_fusion():
    pass

# Comparing the times for subclassed WeldArray (wa), separate object (wne) and
# np.
start = time.time()
arr = simple_loop_binary(wa, NUM_REPS)
arr.eval()
# print(arr)
print("weldarray subclass took {} seconds".format(time.time() - start))

start = time.time()
arr = simple_loop_binary(np, NUM_REPS)
print("numpy ndarray took {} seconds".format(time.time() - start))
