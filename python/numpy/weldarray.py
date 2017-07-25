import numpy as np
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
import time

'''
TODO:
    4. Minor bugs - like reinitializing weld arrays messes it up.
    5. Simplify _build_computation?
    6. Add testing code - simple loop + blackscholes. First do the minimum to
    run these, and then make other improvements.
'''

def _to_symbol(sym):
    return 'sym' + str(sym)

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
    _next_id = 0
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
        obj = np.asarray(input_array).view(cls)
        obj.id = WeldArray._next_id
        WeldArray._next_id += 1

        obj.supported_dtypes = get_supported_types()
        assert str(input_array.dtype) in obj.supported_dtypes
        obj._weld_type = obj.supported_dtypes[str(input_array.dtype)]
        # CHECK: Should this be incremented in __new__ as well?
        obj.id = WeldArray._next_id
        WeldArray._next_id += 1
        obj.unary_ops = get_supported_unary_ops()
        obj.binary_ops = get_supported_binary_ops()
        # Maps elementwise Ids to expressions
        obj.id_map = {}
        # Maps weldobject names to Ids
        obj.name_to_id = {}
        # name of concrete object, if any
        obj.name = None
        obj.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        if isinstance(input_array, WeldArray):
            # need to update the internal state of the weld array
            obj._update_arr(input_array)
        elif isinstance(input_array, np.ndarray):
            obj._update_vector(input_array)
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
            # FIXME: Copy all attributes to self from object - somehow ensuring that
            # the properties of it being a view are preserved.
            pass

    def _update_arr(self, arr):
        self.id_map.update(arr.id_map)
        self.name_to_id.update(arr.name_to_id)
        self.weldobj.update(arr.weldobj)

    def _update_vector(self, vector):
        self.name = self.weldobj.update(vector, WeldVec(WeldFloat()))
        assert self.name not in self.name_to_id
        self.name_to_id[self.name] = self.id

    def __repr__(self):
        '''
        '''
        return super(WeldArray, self).__repr__()

    def __str__(self):
        '''
        Evaluate the array before printing it.
        '''
        # FIXME: Should we evaluate it first?
        # self = self.eval()
        return super(WeldArray, self).__str__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        FIXME: Not dealing with method at all, not sure what needs to be done
        there either.

        Just overwrite the action of weld supported functions and for others, simply,
        use numpy's implementation.
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
            if str(input_.dtype) not in self.supported_dtypes:
                supported_dtype = False

        if ufunc.__name__ in self.unary_ops and supported_dtype:
            assert(len(args) == 1)
            return self._unaryop(ufunc)

        elif ufunc.__name__ in self.binary_ops and supported_dtype:
            assert(len(args) == 2)
            # Figure out which of the args is self
            if isinstance(args[0], WeldArray):
                # first arg is WeldArray, so it must be self
                assert args[0].id == self.id
                other_arg = args[1]
            else:
                other_arg = args[0]
                assert args[1].id == self.id

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
            if str(result.dtype) in self.supported_dtypes:
                return WeldArray(result)
            else:
                return result

    def _set_value(self, value):
        '''
        FIXME: don't just allow random text to be added here?
        '''
        assert self.id not in self.id_map
        self.id_map[self.id] = value

    def eval(self):
        '''
        Evalutes the expression.

        Returns a tuple (result, runtime) where result is the result of the
        computation and runtime is the running time in seconds.
        '''
        code = self._build_computation()
        self.weldobj.weld_code = code

        # FIXME: is there sth better than first make it a np array and then convert back
        # to WeldArray?
        arr = self.weldobj.evaluate(WeldVec(self._weld_type))
        return WeldArray(arr)

    def _binary_op(self, other, binop):
        '''
        '''
        newarr = WeldArray(self)
        template = '{target} = {left} {binop} {right};'
        if isinstance(other, np.float32) or isinstance(other, float):
            newarr._set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=str(other) + 'f',
                binop=binop))

        elif isinstance(other, WeldArray):
            newarr._update_arr(other)
            newarr._set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=_to_symbol(other.id),
                binop=binop))

        # Note: This will be true for WeldArrays too, but because we already
        # checked for weldarray, it must be ndarray.
        elif isinstance(other, np.ndarray):
            assert other.shape == self.shape
            newarr._update_vector(other)
            newarr._set_value(template.format(
                target=_to_symbol(newarr.id),
                left=_to_symbol(self.id),
                right=_to_symbol(newarr.name_to_id[newarr.name]),
                binop=binop))
            # Evaluate all the lazily stored ops, as the caller probably
            # expects to get back a ndarray - which would have latest values.
            newarr = newarr.eval()
        else:
            raise ValueError('invalid type for other')

        return newarr

    def _unaryop(self, func):
        '''
        Create a new array, whose values we will update using weld code.
        FIXME: If someone repeatedly creates arrays, then using self.id here
        fails as previous guys id is no longer valid. Can fix it by using a new
        variable to track ids etc.

        @func: one of self.unary ops.
        '''
        unop = func.__name__
        newarr = WeldArray(self)
        template = '{target} = {unop}({self});'
        newarr._set_value(template.format(
            target=_to_symbol(newarr.id),
            unop=unop,
            self=_to_symbol(self.id)))
        return newarr

    def _build_computation(self):
        '''
        FIXME: This should potentially be changed - and some of the loop fusion
        optimizations should happen in the weld code instead?

        Converts the list of elementwise operations into a
        map(zip(..)) over the data this WeldArray tracks.
        '''
        start = time.time()
        # Number of zipped vectors.
        num_zipped = len(self.name_to_id)
        # Sorted names (this is the order in which things are zipped)
        names = sorted(self.name_to_id.keys())

        # Short circuit if no computation is registered.
        if len(self.id_map) == 0:
            if num_zipped == 0:
                return ''
            elif num_zipped == 1:
                return names[0]
            else:
                raise ValueError('Incosistent state: zip with no ops')

        # FIXME: Need to make this general for all the types supported by Weld
        if num_zipped > 1:
            zip_type = '{' + '{type},'.format(type=self._weld_type.__str__()) * num_zipped
            zip_type = zip_type[:-1] + '}'
        else:
            zip_type = self._weld_type.__str__()

        # (id, instruction) sorted by ID.
        values = sorted([(id, self.id_map[id]) for id in self.id_map])
        values = [(i, 'let ' + v) for i, v in values]

        last_symbol = _to_symbol(values[-1][0])

        computation = '\n'.join([symname for (_, symname) in values] +[last_symbol])
        computation += '\n'

        if num_zipped > 1:
            names_zipped = 'zip(' + ','.join(names) + ')'
        else:
            names_zipped = names[0]

        if num_zipped > 1:
            symdefs = ['let {symstr} = z.${sym};\n'.format(
                sym=i, # Fix me needs to be the zip index
                symstr=_to_symbol(self.name_to_id[id])) for (i, id) in enumerate(names)]
            symdefs = ''.join(symdefs)
        else:
            symdefs = 'let {symstr} = z;\n'.format(
                    symstr=_to_symbol(self.name_to_id[names[0]]))


        final_template = 'map({zipped}, |z: {type}| \n{symdefs}\n{computation})'

        code = final_template.format(
                zipped=names_zipped,
                type=zip_type,
                symdefs=symdefs,
                computation=computation)

        end = time.time()
        print('build computation took: {} seconds'.format(end-start))
        return code

def array(arr, *args, **kwargs):
    '''
    Wrapper around WeldArray - first create np.array and then convert to
    WeldArray.
    '''
    return WeldArray(np.array(arr, *args, **kwargs))
