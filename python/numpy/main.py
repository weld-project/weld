from weldarray import WeldArray
import numpy as np # numpy 1.13 required
import random

test = []
test2 = []
for i in range(10):
    test.append(float(random.random()))
    test2.append(float(random.random()))

test = np.array(test, dtype="float32")
test2 = np.array(test, dtype="float32")

w = WeldArray(test)
w2 = WeldArray(test2)

w3 = np.exp(w)
w3 = w3.eval()
print(w3)


