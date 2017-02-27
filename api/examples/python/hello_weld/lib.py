import os
import sys

home = os.environ.get("WELD_HOME")
if home is None:
    home = "../../../python/"
if home[-1] != "/":
    home += "/"

libpath = home + "api/python"
sys.path.append(libpath)

from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder

import ctypes

import numpy as np

_encoder = NumpyArrayEncoder()
_decoder = NumpyArrayDecoder()

def add(vector, number):
    template = "map({0}, |e| e + {1})"
    obj = WeldObject(_encoder, _decoder)
    if isinstance(vector, WeldObject):
        obj.update(vector, WeldVec(WeldI32()))
        obj.weld_code = template.format(vector.weld_code, str(number))
    else:
        name = obj.update(vector, WeldVec(WeldI32()))
        obj.weld_code = template.format(name, str(number))
    return obj

# Small Test program.

v = np.array(range(1000000), dtype='int32')
v = add(v, 1)
v = add(v, 5)
v = add(v, 10)
print v.weld_code

# Evaluate as a numpy array
print v.evaluate(WeldVec(WeldI32()))


