import numpy as np
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
import time

'''
TODO:
    1. Improve naming scheme - especially for internal variables etc.
    2. Improve readability of the emitted weld code (e.g., binop)
    3. deal with FIXME's
'''

# Temporary debugging aid
DEBUG = False

def get_supported_binary_ops():
    '''
    Returns a dictionary of the Weld supported binary ops, with values being
    their Weld symbol.
    '''
    # TODO: Add inplace ops (__iadd__) and stuff.
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

class WeldArray(np.ndarray):
    '''
    '''
    def __new__(cls, input_array, *args, **kwargs):
        '''
        @input_array: original ndarray from which the new array is derived.
        '''
        assert isinstance(input_array, np.ndarray), 'only support ndarrays'
        # FIXME: When initializing WeldArray from an old WeldArray, should we
        # copy it to a new place - or just using view is ok? Don't see any
        # performance difference for now, but might need to consider this
        # further.
        # if isinstance(input_array, WeldArray):
            # # Not sharing same memory.
            # obj = np.array(input_array, subok=True)
        # elif isinstance(input_array, np.ndarray):
            # # Sharing same memory.
            # obj = np.asarray(input_array).view(cls)

        # sharing memory with the ndarray/weldarry
        obj = np.asarray(input_array).view(cls)

        # this var id list is maintained per WeldArray to name the variables as
        # we register more ops to the WeldArray.
        obj._var_id = 0

        # dicts that map the ops name to its weld IR equivalent.
        obj._supported_dtypes = get_supported_types()
        assert str(input_array.dtype) in obj._supported_dtypes
        obj._weld_type = obj._supported_dtypes[str(input_array.dtype)]
        obj.unary_ops = get_supported_unary_ops()
        obj.binary_ops = get_supported_binary_ops()

        # name refers to the name for the given WeldArray so weldobj can
        # interpret it when we emit the IR.
        obj.name = None
        obj.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        obj.weldobj.weld_code = ''

        if isinstance(input_array, WeldArray):
            # need to update the internal state of the weld array
            obj._update_weldarray(input_array, True)
        elif isinstance(input_array, np.ndarray):
            obj._update_ndarray(input_array, True)
        else:
            return NotImplemented

        return obj

    def __array_finalize__(self, obj):
        '''
        FIXME: This gets called both when slicing / or creating a new array.
        At least need to deal with slicing here...
        '''
        if obj is None:
            return
        if isinstance(obj, WeldArray):
            # FIXME: Are there cases besides views when we might be here?
            pass

        # TODO: Might want to move some of the general initialization stuff
        # here from __new__?

    def _update_weldarray(self, arr, new):
        '''
        Need to store the names from the array to this one.
        @new: If this is being called from __new__. Otherwise, won't update
        name/_var_id of self. It is also called internally whenever applying a
        binary op to a given vec.
        '''
        self.weldobj.update(arr.weldobj)
        # update whatever weld code the previous array has built so far.
        # Need to append to the previous code because the given array might
        # already have code to execute.
        self.weldobj.weld_code += arr.weldobj.weld_code
        if new:
            self._var_id = arr._var_id
            self.name = arr.name

    def _update_ndarray(self, vector, new):
        '''
        vector: ndarray.
        new: if being called from __new__ while generating a new array.
        '''
        assert isinstance(vector, np.ndarray)
        name = self.weldobj.update(vector, self._supported_dtypes[str(vector.dtype)])
        # Initialize first tmp variable = name. This is required because
        # _binary_op expects vector name of type eij to compute on.
        tmp_var = name + str(self._var_id)
        template = 'let {tmp_var} = {vec};\n'
        self.weldobj.weld_code += template.format(tmp_var=tmp_var, vec=name)
        if new:
            self.name = name
            self._var_id += 1

    def __repr__(self):
        '''
        '''
        return super(WeldArray, self).__repr__()

    def __str__(self):
        '''
        Evaluate the array before printing it / or just print it directly?
        '''
        # FIXME: Should we evaluate it first? Will need to figure out how
        # change the values of the current WeldArray so that in future calls it
        # doesn't need to execute eval again.
        return super(WeldArray, self).__str__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        FIXME: Not dealing with method at all, not sure what needs to be done
        there either.

        Just overwrite the action of weld supported functions and for others,
        pass on to numpy's implementation.

        TODO: add input descriptions.
        TODO: Deal with outputs (should solve the issue with a += b too)
        '''
        # Collect the input args
        args = []
        supported_dtype = True
        for input_ in inputs:
            args.append(input_)
            if not isinstance(input_, np.ndarray):
                # For instance, it can be just a number
                # FIXME: Add more checks for its type here.
                continue
            if str(input_.dtype) not in self._supported_dtypes:
                supported_dtype = False

        if ufunc.__name__ in self.unary_ops and supported_dtype:
            assert(len(args) == 1)
            return self._unary_op(self.unary_ops[ufunc.__name__])

        elif ufunc.__name__ in self.binary_ops and supported_dtype:
            assert(len(args) == 2)
            # Figure out which of the args is self
            if isinstance(args[0], WeldArray):
                # first arg is WeldArray, so it must be self
                assert args[0].name == self.name
                other_arg = args[1]
            else:
                other_arg = args[0]
                assert args[1].name == self.name

            return self._binary_op(other_arg, self.binary_ops[ufunc.__name__])

        else:
            # For implementation reasons, we need to convert all args to
            # ndarray before we can call super's implementation on these.
            for i, arg_ in enumerate(args):
                if isinstance(arg_, WeldArray):
                    # Evaluate all the lazily stored computations of the
                    # WeldArray before applying the super method.
                    arg_ = arg_.eval()
                    args[i] = arg_.view(np.ndarray)

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
                                                     *args, **kwargs)
            if str(result.dtype) in self._supported_dtypes:
                return WeldArray(result)
            else:
                return result

    def eval(self):
        '''
        Evalutes the expression based on weldobj.weld_code.
        Returns a new weld array based on the computation.
        '''
        # FIXME: is there sth better than first make it a np array and then convert back
        # to WeldArray?
        # add a return variable line to the end of the weld_code we have.
        if self._var_id > 0:
            ret_var = self.name + str(self._var_id-1)
        else:
            # no ops have been registered with this array so far.
            return WeldArray(self)

        old_code = self.weldobj.weld_code
        self.weldobj.weld_code += ret_var + '\n'
        if DEBUG: print('code in eval: ', self.weldobj.weld_code)

        arr = self.weldobj.evaluate(WeldVec(self._weld_type))
        # FIXME: Temporary Solution:
        # In case some one wants to call eval again, we should remove the last
        # line. e.g., np.isfinite(a) called by some library - would induce eval
        # being called in __ufunc__, and then future calls to eval would fail.
        # Better Solution: Not sure. We could update the array in place, but
        # that may not be possible with multi dimensional arrays.
        self.weldobj.weld_code = old_code

        return WeldArray(arr)

    def _unary_op(self, unop):
        '''
        Create a new array, and updates weldobj.weld_code with the code for
        unop.
        @unop: str, weld IR for the given function.
        '''
        newarr = WeldArray(self)
        template = 'let {tmp_name} = map({arr}, |z: {weld_type}| {unop}(z));\n'
        code = template.format(tmp_name = self.name + str(self._var_id),
                arr=self.name + str(self._var_id-1),
                weld_type=self._weld_type.__str__(),
                unop=unop)
        newarr._var_id += 1
        newarr.weldobj.weld_code += code

        return newarr

    def _binary_op(self, other, binop):
        '''
        @other: ndarray, weldarray or number.
        FIXME: Handling the case when the types of the arrays don't match.
        '''
        # @arr{i}: latest array based on the ops registered on operand{i}. {{
        # is used for escaping { for format.
        template = 'let {tmp_name} = map(zip({arr1},{arr2}), |z: \
            {{{type1},{type2}}}| z.$0 {binop} z.$1);\n'
        result = WeldArray(self)
        is_weld_array = False

        # FIXME: support other dtypes.
        if isinstance(other, np.float32) or isinstance(other, float):
            template = 'let {tmp_name} = map({arr1}, |z: \
                    {type}| z {binop} {other}f);\n'
            code = template.format(tmp_name = result.name + str(result._var_id),
                        arr1 = result.name + str(result._var_id-1),
                        type = result._weld_type.__str__(),
                        binop = binop,
                        other = str(other))
            result._var_id += 1
            result.weldobj.weld_code += code
            if DEBUG: print('returning with code: ', self.weldobj.weld_code)
            return result

        elif isinstance(other, WeldArray):
            # Let's add all the weld_code/vars from the second array to this.
            result._update_weldarray(other, False)
            is_weld_array = True

        # Note: This will be true for WeldArrays too, but because we already
        # checked for weldarray, it must be ndarray.
        elif isinstance(other, np.ndarray):
            # turn it into WeldArray - for convenience, as will have access to
            # the names assigned to the given vector etc then for finishing
            # the code template.
            other = WeldArray(other)
            result._update_weldarray(other, False)
        else:
            raise ValueError('invalid type for other in binaryop')

        code = template.format(tmp_name = self.name + str(result._var_id),
                arr1 = self.name + str(result._var_id-1),
                arr2 = other.name + str(other._var_id-1),
                type1 = result._weld_type.__str__(),
                type2 = other._weld_type.__str__(),
                binop = binop)

        result.weldobj.weld_code += code
        if DEBUG: print('final weld code = ', result.weldobj.weld_code)
        result._var_id += 1

        # if other was an ndarray then evaluate everything so far.
        if not is_weld_array:
            # Evaluate all the lazily stored ops, as the caller probably
            # expects to get back a ndarray - which would have latest values.
            # newarr = newarr.eval()
            result = result.eval()

        return result


def array(arr, *args, **kwargs):
    '''
    Wrapper around WeldArray - first create np.array and then convert to
    WeldArray.
    '''
    return WeldArray(np.array(arr, *args, **kwargs))
