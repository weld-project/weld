
from lib import *
import numpy as np

import time

# Input data size
size = (2 << 28)

a = np.zeros((size), dtype='int32')

print "Running NumPy..."
start = time.time()
for i in range(40):
    a += i
end = time.time()
print "NumPy result:", a.sum()
np_time = (end - start)
print "({:.3} seconds)".format(np_time)

# Reset
a = np.zeros((size), dtype='int32')

print "Running Weld..."
v = HelloWeldVector(a)
start = time.time()
for i in range(40):
    v += i
print "Weld result:", v.sum()
end = time.time()
weld_time = (end - start)
print "({:.3} seconds)".format(weld_time)

# Print speedup/slowdown.
speedup = (np_time / weld_time)
print "{:.3}x faster".format((np_time / weld_time))

print ""
print ""

assert speedup > 1000.0, "Haven't hit 1000x yet!"
