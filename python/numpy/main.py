from weldarray import WeldArray
# from weld_numpy_elemwise import *
import numpy as np # numpy 1.13 required
import random

test = []
test2 = []
for i in range(10):
    test.append(float(random.random()))
    test2.append(float(random.random()))

test = np.array(test, dtype="float32")
print("test = ", test)
test2 = np.array(test, dtype="float32")

w = WeldArray(test)
print("w constructed")

# w2 = w[2:4]
# print(type(w2))
# w2 = WeldArray(test2)
w3 = np.log(w)
# w3 = w3.eval()
print(w3)
w3.eval()



