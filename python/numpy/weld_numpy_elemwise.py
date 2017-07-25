#
# An example library that uses Weld
#
# This library operates on NumPy ndarrays, and emulates one dimensional float vectors.
#

# import ctypes # TODO: What is this used for.
import os
import sys

import copy

import numpy as np
import scipy.special as ss

from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder

all_ops = set(["log", "sqrt", "exp", "erf", "+", "-", "/", "*"])
implemented = copy.deepcopy(all_ops)
# implemented = set(["exp", "+", "-", "/", "*"])


# Total NVL runtime accumulated.
_accum_time = [0.0, 0]

def _to_symbol(sym):
    return "sym" + str(sym)

def _check_ndarray(arr):
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == "float32"

def reset_time():
    """
    Resets the running time.
    """
    _accum_time = [0.0, 0]

def read_time():
    """
    Returns the total time spent executing NVL code since the
    last time `reset_time` was called.
    """
    return _accum_time[0]

def read_time_count():
    """
    Returns the total number of times the counter is updated.
    """
    return _accum_time[1]

def implement_operator(op):
    """
    Set `op` to "implemented" (so it runs with NVL).
    The op must be in the list of all supported ops.
    """
    assert op in all_ops
    if op not in implemented:
        implemented.add(op)

def remove_operator(op):
    """
    Removes `op` from the list of implemented operators.
    """
    assert op in all_ops
    try:
        implemented.remove(op)
    except KeyError:
        pass

def supported_operators():
    """
    Returns a list of all supported operators
    """
    return list(all_ops)

def implemented_operators():
    """
    Returns a list of all currently implemented operators
    """
    return list(implemented)

def array(arr, *args, **kwargs):
    '''
    Wrapper around WeldArray - first create np.array and then convert to
    WeldArray.
    '''
    return WeldArray(np.array(arr, *args, **kwargs))

# def array(arr):
    # """
    # returns an NVL array capturing arr.
    # """
    # if isinstance(arr, list):
        # arr = rawarray(arr)
    # _check_ndarray(arr)
    # return WeldArray(arr)

# def rawarray(arr):
    # """
    # Returns a "raw" (Numpy) array. wrapping a python array.
    # """
    # return np.array(arr, dtype="float32")

def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    return arr._to_numpy()

def _unop_impl_prefix(obj, unop):
    newarr = WeldArray(obj)
    template = "{target} = {unop}({self});"
    unop_code = template.format(
        target=_to_symbol(newarr.id),
        unop=unop,
        self=_to_symbol(obj.id))

    newarr.set_value(unop_code)

    return newarr

def log(obj):
    if isinstance(obj, np.ndarray):
        return np.log(obj)
    if "log" in implemented:
        return _unop_impl_prefix(obj, "log")
    else:
        arr = obj._to_numpy()
        start = time.time()
        arr = np.log(arr)
        end = time.time()
        _accum_time[0] += (end - start)
        _accum_time[1] += 1
        return array(arr)

def exp(obj):
    if isinstance(obj, np.ndarray):
        return np.exp(obj)
    if "exp" in implemented:
        return _unop_impl_prefix(obj, "exp")
    # CHECK: Point of the else.
    else:
        arr = obj._to_numpy()
        start = time.time()
        arr = np.exp(arr)
        end = time.time()
        _accum_time[0] += (end - start)
        _accum_time[1] += 1
        return array(arr)

def sqrt(obj):
    if isinstance(obj, np.ndarray):
        return np.sqrt(obj)
    if "sqrt" in implemented:
        return _unop_impl_prefix(obj, "sqrt")
    else:
        arr = obj._to_numpy()
        start = time.time()
        arr = np.sqrt(arr)
        end = time.time()
        _accum_time[0] += (end - start)
        _accum_time[1] += 1
        return array(arr)

def erf(obj):
    # If we pass in a numpy thing.
    if isinstance(obj, np.ndarray):
        return np.erf(obj)
    if "erf" in implemented:
        return _unop_impl_prefix(obj, "erf")
    else:
        # why is this required?
        arr = obj._to_numpy()
        start = time.time()
        arr = ss.erf(arr)
        end = time.time()
        _accum_time[0] += (end - start)
        _accum_time[1] += 1
        return array(arr)

# Not sure in this case how we would implement things like getitem etc - would
# be more complicated as we only have reference to WeldObject and not the np
# array.

class WeldArray(object):
    _next_id = 0
    def __init__(self, input_array):
        # A Limited library that only supports 1D float vectors.
        self.id = WeldArray._next_id
        WeldArray._next_id += 1

        # Maps elementwise Ids to expressions
        self.id_map = {}
        # Maps weldobject names to Ids
        self.name_to_id = {}
        # name of concrete object, if any
        self.name = None

        # equivalent of weldobj
        self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())

        if isinstance(input_array, np.ndarray):
            _check_ndarray(input_array)
            self.update_vector(input_array)
        elif isinstance(input_array, WeldArray):
            self.update_arr(input_array)

    # CHECK: so basically sets the value of the current array to some
    # code/operation that is to be run on it. When is this actually run? Is
    # this because of lazy evaluation? Happens in build computation graph?
    def set_value(self, value):
        assert self.id not in self.id_map
        self.id_map[self.id] = value

    def update_vector(self, vector):
        self.name = self.weldobj.update(vector, WeldVec(WeldFloat()))
        assert self.name not in self.name_to_id
        self.name_to_id[self.name] = self.id

    def update_arr(self, arr):
        self.id_map.update(arr.id_map)
        self.name_to_id.update(arr.name_to_id)
        self.weldobj.update(arr.weldobj)

    def build_computation(self):
        """
        Converts the list of elementwise operations into a
        map(zip(..)) over the data this WeldArray tracks.
        """
        # Number of zipped vectors.
        num_zipped = len(self.name_to_id)
        # Sorted names (this is the order in which things are zipped)
        names = sorted(self.name_to_id.keys())

        # Short circuit if no computation is registered.
        if len(self.id_map) == 0:
            if num_zipped == 0:
                return ""
            elif num_zipped == 1:
                return names[0]
            else:
                raise ValueError("Incosistent state: zip with no ops")

        if num_zipped > 1:
            zip_type = "{" + "f32," * num_zipped
            zip_type = zip_type[:-1] + "}"
        else:
            zip_type = "f32"

        # (id, instruction) sorted by ID.
        values = sorted([(id, self.id_map[id]) for id in self.id_map])
        values = [(i, "let " + v) for i, v in values]

        last_symbol = _to_symbol(values[-1][0])

        computation = "\n".join([symname for (_, symname) in values] +[last_symbol])
        computation += "\n"

        if num_zipped > 1:
            names_zipped = "zip(" + ",".join(names) + ")"
        else:
            names_zipped = names[0]

        if num_zipped > 1:
            symdefs = ["let {symstr} = z.${sym};\n".format(
                sym=i, # Fix me needs to be the zip index
                symstr=_to_symbol(self.name_to_id[id])) for (i, id) in enumerate(names)]
            symdefs = "".join(symdefs)
        else:
            symdefs = "let {symstr} = z;\n".format(
                    symstr=_to_symbol(self.name_to_id[names[0]]))


        final_template = "map({zipped}, |z: {type}| \n{symdefs}\n{computation})"

        code = final_template.format(
                zipped=names_zipped,
                type=zip_type,
                symdefs=symdefs,
                computation=computation)
        return code

    def _to_numpy(self):
        """
        Evalutes the expression.

        Returns a tuple (result, runtime) where result is the result of the
        computation and runtime is the running time in seconds.
        """
        code = self.build_computation()
        self.weldobj.weld_code = code
        arr = self.weldobj.evaluate(WeldVec(WeldFloat()))

        # FIXME: Find equivalent of returned t values from evaluate
        # _accum_time[0] += t
        #_accum_time[1] += 1
        return arr

    def _binop_impl_infix(self, other, binop):
        newarr = WeldArray(self)
        template = "{target} = {left} {binop} {right};"
        if isinstance(other, np.float32) or isinstance(other, float):
            newarr.set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=str(other) + "f",
                binop=binop))

        elif isinstance(other, np.ndarray):
            _check_ndarray(other)
            newarr.update_vector(other)
            newarr.set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=_to_symbol(newarr.name_to_id[newarr.name]),
                binop=binop))
        elif isinstance(other, WeldArray):
            newarr.update_arr(other)
            newarr.set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=_to_symbol(other.id),
                binop=binop))
        else:
            raise ValueError("invalid type for other")

        return newarr

    def _unop_impl_prefix(self, unop):

        # Create a new array, whose values we will update using weld code.
        newarr = WeldArray(self)
        template = "{target} = {unop}({self});"
        newarr.set_value(template.format(
            target=_to_symbol(newarr.id),
            unop=unop,
            self=_to_symbol(self.id)))
        return newarr

    def add(self, other):
        if "+" not in implemented:
            arr = self._to_numpy()
            if isinstance(other, WeldArray):
                other = other._to_numpy()
            start = time.time()
            arr = arr + other
            end = time.time()
            _accum_time[0] += (end - start)
            _accum_time[1] += 1
            return array(arr)
        return self._binop_impl_infix(other, "+")

    def __add__(self, other):
        return self.add(other)
    def __iadd__(self, other):
        return self.add(other)
    def __radd__(self, other):
        return self.add(other)

    def sub(self, other):
        if "-" not in implemented:
            arr = self._to_numpy()
            if isinstance(other, WeldArray):
                other = other._to_numpy()
            start = time.time()
            arr = arr - other
            end = time.time()
            _accum_time[0] += (end - start)
            _accum_time[1] += 1
            return array(arr)
        return self._binop_impl_infix(other, "-")

    def __sub__(self, other):
        return self.sub(other)
    def __isub__(self, other):
        return self.sub(other)
    def __rsub__(self, other):
        return self.neg().sub(-other)

    def mul(self, other):
        if "*" not in implemented:
            arr = self._to_numpy()
            if isinstance(other, WeldArray):
                other = other._to_numpy()
            start = time.time()
            arr = arr * other
            end = time.time()
            _accum_time[0] += (end - start)
            _accum_time[1] += 1
            return array(arr)
        return self._binop_impl_infix(other, "*")

    def __mul__(self, other):
        return self.mul(other)
    def __imul__(self, other):
        return self.mul(other)
    def __rmul__(self, other):
        return self.mul(other)

    def div(self, other):
        if "/" not in implemented:
            arr = self._to_numpy()
            if isinstance(other, WeldArray):
                other = other._to_numpy()
            start = time.time()
            arr = arr / other
            end = time.time()
            _accum_time[0] += (end - start)
            _accum_time[1] += 1
            return array(arr)
        return self._binop_impl_infix(other, "/")

    def __div__(self, other):
        return self.div(other)
    def __idiv__(self, other):
        return self.div(other)

    def neg(self):
        if "-" not in implemented:
            arr = self._to_numpy()
            start = time.time()
            arr = -arr
            end = time.time()
            _accum_time[0] += (end - start)
            _accum_time[1] += 1
            return array(arr)
        return self._unop_impl_prefix("-")
    def __neg__(self):
        return self._unop_impl_prefix("-")
