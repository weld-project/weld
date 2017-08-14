import numpy as np
import sys
sys.path.append('..')
import weldarray as wa
import py.test
import random
import timeit
import time

NUM_REPS = 500
NUM_ELS = 100

def simple_loop(lib, reps):
    '''
    lib = np or wa.
    '''
    np.random.seed(1)
    test = np.random.rand(NUM_ELS)
    test2 = np.random.rand(NUM_ELS)

    arr = lib.array(test, dtype="float32")
    arr2 = lib.array(test2, dtype="float32")

    start = time.time()
    for i in range(reps):
        arr = np.sqrt(arr)
        # arr = arr + arr2

    if isinstance(arr, wa.WeldArray):
        arr = arr.evaluate()

    print("{} took {} seconds".format(type(arr), time.time() - start))

def simple_loop_fusion():
    pass

# Comparing the times for subclassed WeldArray (wa), separate object (wne) and
# np.
arr = simple_loop(wa, NUM_REPS)
arr = simple_loop(np, NUM_REPS)
