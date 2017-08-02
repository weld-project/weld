import numpy as np
import sys
sys.path.append('..')
import weldarray as wa
import py.test
import random
import timeit
import time

NUM_REPS = 100
NUM_ELS = 100

def simple_loop(lib, reps):
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
