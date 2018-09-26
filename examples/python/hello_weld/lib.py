
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder, ScalarDecoder

_encoder = NumpyArrayEncoder()
_decoder = NumpyArrayDecoder()


class HelloWeldVector(object):

    def __init__(self, vector):
        self.vector = vector
        self.weldobj = WeldObject(_encoder, _decoder)
        name = self.weldobj.update(vector, WeldVec(WeldInt()))
        self.weldobj.weld_code = name
        self.cached = None

    def add(self, number):
        self.cached = None
        template = "map({0}, |e| e + {1})"
        self.weldobj.weld_code = template.format(
            self.weldobj.weld_code, str(number))
        return self

    def sum(self):
        self.cached = None
        template = "result(for({0}, merger[i64,+], |b,i,e| merge(b, i64(e))))"
        prev_code = self.weldobj.weld_code
        self.weldobj.weld_code = template.format(self.weldobj.weld_code)
        self.weldobj.decoder = ScalarDecoder()
        result = self.weldobj.evaluate(WeldLong(), verbose=False)
        self.weldobj.decoder = _decoder
        self.weldobj.weld_code = prev_code
        return result

    def __iadd__(self, other):
        return self.add(other)

    def multiply(self, number):
        self.cached = None
        template = "map({0}, |e| e * {1})"
        self.weldobj.weld_code = template.format(
            self.weldobj.weld_code, str(number))
        return self

    def subtract(self, number):
        self.cached = None
        template = "map({0}, |e| e - {1})"
        self.weldobj.weld_code = template.format(
            self.weldobj.weld_code, str(number))
        return self

    def divide(self, number):
        self.cached = None
        template = "map({0}, |e| e / {1})"
        self.weldobj.weld_code = template.format(
            self.weldobj.weld_code, str(number))
        return self

    def __str__(self):
        if self.cached is not None:
            return str(self.cached)
        v = self.weldobj.evaluate(WeldVec(WeldInt()), verbose=False)
        self.cached = v
        return str(v)
