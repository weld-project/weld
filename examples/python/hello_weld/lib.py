import os
import sys
import ctypes
import numpy as np

home = os.environ.get("WELD_HOME")
if home is None:
    home = "../../../python/"
if home[-1] != "/":
    home += "/"

libpath = home + "api/python"
sys.path.append(libpath)

from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder, ScalarDecoder

_encoder = NumpyArrayEncoder()
_decoder = NumpyArrayDecoder()

class HelloWeldVector(object):
    def __init__(self, vector):
        self.vector = vector
        self.weldobj = WeldObject(_encoder, _decoder)
        name = self.weldobj.update(vector, WeldVec(WeldInt()), override=True)
        self.weldobj.weld_code = name
        self.cached = None

    def add(self, number):
        self.cached = None
        template = "map({0}, |e| e + {1})"
        self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
        return self

    def sum(self):
        self.cached = None
        template = "result(for({0}, merger[i64,+], |b,i,e| merge(b, i64(e))))"
        prev_code = self.weldobj.weld_code
        self.weldobj.weld_code = template.format(self.weldobj.weld_code)
        self.weldobj.decoder = ScalarDecoder()
        result = self.weldobj.evaluate(WeldLong())
        self.weldobj.decoder = _decoder
        self.weldobj.weld_code = prev_code
        return result

    def __iadd__(self, other):
        return self.add(other)

    def multiply(self, number):
        self.cached = None
        template = "map({0}, |e| e * {1})"
        self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
        return self

    def subtract(self, number):
        self.cached = None
        template = "map({0}, |e| e - {1})"
        self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
        return self

    def divide(self, number):
        self.cached = None
        template = "map({0}, |e| e / {1})"
        self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
        return self

    def __str__(self):
        if self.cached is not None:
            return str(v)
        v = self.weldobj.evaluate(WeldVec(WeldInt()), verbose=False)
        self.cached = v
        return str(v)

