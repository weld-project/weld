import numpy as np
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
import time

# Temporary debugging aid
DEBUG = False

def addr(arr):
    '''
    returns address of the given ndarray.
    '''
    return arr.__array_interface__['data'][0]

def get_supported_binary_ops():
    '''
    Returns a dictionary of the Weld supported binary ops, with values being
    their Weld symbol.
    '''
    binary_ops = {}
    binary_ops[np.add.__name__] = '+'
    binary_ops[np.subtract.__name__] = '-'
    binary_ops[np.multiply.__name__] = '*'
    binary_ops[np.divide.__name__] = '/'
    return binary_ops

def get_supported_unary_ops():
    '''
    Returns a dictionary of the Weld supported unary ops, with values being
    their Weld symbol.
    '''
    unary_ops = {}
    unary_ops[np.exp.__name__] = 'exp'
    unary_ops[np.log.__name__] = 'log'
    unary_ops[np.sqrt.__name__] = 'sqrt'
    return unary_ops

def get_supported_types():
    '''
    '''
    types = {}
    types['float32'] = WeldFloat()
    types['float64'] = WeldDouble()
    types['int32'] = WeldInt()
    types['int64'] = WeldLong()
    return types

# Global variables for the WeldArray type, used for lookups
BINARY_OPS = get_supported_binary_ops()
UNARY_OPS = get_supported_unary_ops()
SUPPORTED_DTYPES = get_supported_types()

class WeldArray(np.ndarray):
    '''
    '''
    def __new__(cls, input_array, *args, **kwargs):
        '''
        @input_array: original ndarray from which the new array is derived.
        '''
        assert isinstance(input_array, np.ndarray), 'only support ndarrays'
        # sharing memory with the ndarray/weldarry
        obj = np.asarray(input_array).view(cls)
        obj._gen_weldobj(input_array)
        return obj

    def __array_finalize__(self, obj):
        '''
        Gets called whenever a new array is created. This is possible due to:
            - from __new__
            - generating view of weldarray
            - TODO.

        @self: current array.
        @obj: base array.
        In particular, the cases we care about are:
            @self: new, smaller array.
            @obj: WeldArray.

        In case of init WeldArray (from WeldArray, or numpy array):
            @self: new weldarray
            @obj: base ndarray.
        '''
        if obj is None:
            return

        # dicts that map the ops name to its weld IR equivalent.
        assert str(obj.dtype) in SUPPORTED_DTYPES
        self._weld_type = SUPPORTED_DTYPES[str(obj.dtype)]

        if isinstance(obj, WeldArray):
            # evaluate the parent array (obj) and update values of self.
            parent = obj.eval()
            start = (addr(self) - addr(obj)) / self.itemsize
            end = start + len(self)
            # TODO: optimize this routine.
            for i in range(end-start):
                self[i] = parent[start+i]

            # Create weldobj for self, treating the array self as ndarray.
            self._gen_weldobj(self.view(np.ndarray))

    def _gen_weldobj(self, arr):
        '''
        Generating a new WeldArray from a given arr.
        @arr: WeldArray or ndarray.

        Sets self.name and self.weldobj.
        '''
        self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        if isinstance(arr, WeldArray):
            self.weldobj.update(arr.weldobj)
            self.weldobj.weld_code = arr.weldobj.weld_code
            self.name = arr.name
        elif isinstance(arr, np.ndarray):
            # weldobj returns the name bound to the given array. That is also
            # the array that future ops will act on, so set weld_code to it.
            self.name = self.weldobj.weld_code = self.weldobj.update(arr,
                    SUPPORTED_DTYPES[str(arr.dtype)])
        else:
            assert False, '_gen_weldobj has unsupported input array'

    def __repr__(self):
        '''
        '''
        return super(WeldArray, self).__repr__()

    def __str__(self):
        '''
        Evaluate the array before printing it / or just print it directly?
        FIXME: Evaluate it before printing.
        '''
        return super(WeldArray, self).__str__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Just overwrite the action of weld supported functions and for others,
        pass on to numpy's implementation.

        FIXME: Not dealing with method at all. Check use cases.
        TODO: add input descriptions.
        '''
        input_args = [inp for inp in inputs]
        arrays = all(isinstance(i, np.ndarray) for i in input_args)
        supported = True
        if arrays:
            supported = all(str(_inp.dtype) in SUPPORTED_DTYPES for _inp in input_args)
            if len(input_args) == 2 and input_args[0].dtype != input_args[1].dtype:
                supported = False
        else:
            # FIXME: add scalar tests
            pass
        outputs = kwargs.pop('out', None)
        if outputs:
            if len(outputs) == 1:
                output = outputs[0]
            else:
                supported = False
        else:
            output = None

        if ufunc.__name__ in UNARY_OPS and supported:
            assert(len(input_args) == 1)
            return self._unary_op(UNARY_OPS[ufunc.__name__], result=output)

        if ufunc.__name__ in BINARY_OPS and supported:
            assert(len(input_args) == 2)
            # self can be first or second arg.
            if isinstance(input_args[0], WeldArray):
                # first arg is WeldArray, must be self
                assert input_args[0].name == self.name
                other_arg = input_args[1]
            else:
                other_arg = input_args[0]
                assert input_args[1].name == self.name

            return self._binary_op(other_arg, BINARY_OPS[ufunc.__name__],
                    result=output)

        # Relegating the work to numpy.
        # For implementation reasons, we need to convert all input_args to
        # ndarray before we can call super's implementation on these.
        for i, arg_ in enumerate(input_args):
            if isinstance(arg_, WeldArray):
                # Evaluate all the lazily stored computations first.
                arg_ = arg_.eval()
                input_args[i] = arg_.view(np.ndarray)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, WeldArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        result = super(WeldArray, self).__array_ufunc__(ufunc, method,
                                                 *input_args, **kwargs)
        if str(result.dtype) in SUPPORTED_DTYPES:
            return WeldArray(result)
        else:
            return result

    def eval(self):
        '''
        Evalutes the expression based on weldobj.weld_code.
        Returns a new weld array based on the computation.
        '''
        if self.weldobj.weld_code == self.name:
            # no ops have been registered with this array so far.
            return WeldArray(self)
        if DEBUG: print('code len in eval: ', len(self.weldobj.weld_code))
        if DEBUG: print('code in eval: ', self.weldobj.weld_code)

        arr = self.weldobj.evaluate(WeldVec(self._weld_type), verbose=True)

        # Now that the evaluation is done - create new weldobject for self,
        # initalized from the given arr.
        self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        self.weldobj.weld_code = ''
        self.name = self.weldobj.update(arr, SUPPORTED_DTYPES[str(arr.dtype)])
        self.weldobj.weld_code = self.name

        return WeldArray(arr)

    def _unary_op(self, unop, result=None):
        '''
        Create a new array, and updates weldobj.weld_code with the code for
        unop.
        @unop: str, weld IR for the given function.
        '''
        if result is None:
            result = WeldArray(self)

        template = 'map({arr}, |z : {type}| {unop}(z))'
        code = template.format(arr = result.weldobj.weld_code,
                type=result._weld_type.__str__(),
                unop=unop)
        result.weldobj.weld_code = code
        return result

    def _binary_op(self, other, binop, result=None):
        '''
        @other: ndarray, weldarray or number.
        '''
        if result is None:
            result = WeldArray(self)

        is_weld_array = False
        # FIXME: support other dtypes.
        if isinstance(other, np.float32) or isinstance(other, float):
            template = 'map({arr}, |z: {type}| z {binop} {other}f)'
            code = template.format(arr = result.weldobj.weld_code,
                        type = result._weld_type.__str__(),
                        binop = binop,
                        other = str(other))
            result.weldobj.weld_code = code
            return result

        elif isinstance(other, WeldArray):
            is_weld_array = True
        elif isinstance(other, np.ndarray):
            # turn it into WeldArray - for convenience, as this reduces to the
            # case of other being a WeldArray.

            # FIXME: To optimize this, might want to update the weldobj without
            # converting other to WeldArray.
            other = WeldArray(other)
        else:
            raise ValueError('invalid type for other in binaryop')

        result.weldobj.update(other.weldobj)
        # @arr{i}: latest array based on the ops registered on operand{i}. {{
        # is used for escaping { for format.
        template = 'map(zip({arr1},{arr2}), |z: {{{type1},{type2}}}| z.$0 {binop} z.$1)'

        code = template.format(arr1 = result.weldobj.weld_code,
                arr2 = other.weldobj.weld_code,
                type1 = result._weld_type.__str__(),
                type2 = other._weld_type.__str__(),
                binop = binop)

        result.weldobj.weld_code = code

        if not is_weld_array:
            if DEBUG: print("not weld array")
            # Evaluate all the lazily stored ops, as the caller probably
            # expects to get back a ndarray - which would have latest values.
            result = result.eval()

        return result


def array(arr, *args, **kwargs):
    '''
    Wrapper around WeldArray - first create np.array and then convert to
    WeldArray.
    '''
    return WeldArray(np.array(arr, *args, **kwargs))
