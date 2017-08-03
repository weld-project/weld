from weld.weldobject import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
from weldnumpy import *

class WeldArray(np.ndarray):
    '''
    A new WeldArray can be created in three ways:
        - Explicit constructor. This is dealt with in __new__
        - Generating view of WeldArray (same as ndarray)
        - fancy indexing (same as ndarray)
    The second and third case does not pass through __new__. Instead, numpy
    guarantees that after initializing an array, in all three methods, numpy
    will call __array_finalize__. But we are able to deal with the 2nd and 3rd
    case in __getitem__ so there does not seem to be any need to use
    __array_finalize (besides __array_finalize__ also adds a function call to
    the creation of a new array, which adds to the overhead as compared to
    numpy for initializing arrays.
    '''
    def __new__(cls, input_array, verbose=False, *args, **kwargs):
        '''
        @input_array: original ndarray from which the new array is derived.
        '''
        assert isinstance(input_array, np.ndarray), 'only support ndarrays'
        assert str(input_array.dtype) in SUPPORTED_DTYPES

        # sharing memory with the ndarray/weldarry
        obj = np.asarray(input_array).view(cls)
        obj._gen_weldobj(input_array)
        obj._weld_type = SUPPORTED_DTYPES[str(input_array.dtype)]
        obj.verbose = verbose
        # Views
        obj.view_parents = []
        obj.view_children = []

        return obj

    def __repr__(self):
        '''
        Evaluate, and then let ndarray deal with it.
        '''
        arr = self._eval()
        return arr.__repr__()

    def __str__(self):
        '''
        Evaluate the array before printing it.
        '''
        arr = self._eval()
        return arr.__str__()

    def __getitem__(self, idx):
        '''
        a[...] type of indexing.
        Deal with the following three scenarios:
            - idx is a scalar
            - idx is a slice (e.g., 3:5)
            - idx is an array (fancy indicing)

        Steps:
            1. Evaluate current value.
            2. Call np.ndarray on the evaluated array.
            3. Convert back to WeldArray if needed.
        '''
        # evaluate all lazily stored ops.
        arr = self._eval()
        # arr is an ndarray, so we call ndarray's implementation on it.
        res = arr.__getitem__(idx)

        if isinstance(idx, slice):
            # is res really a view of arr?
            if is_view_child(res, arr):
                res = WeldArray(res)
                # update both the lists.
                self.view_children.append(res)
                res.view_parents.append(self)
                # if self already has parents
                for p in self.view_parents:
                    # parenting is transitive.
                    res.view_parents.append(p)

                return res

        elif isinstance(idx, np.ndarray):
            # this will not have shared memory with parent array.
            return WeldArray(res)
        else:
            # return the scalar.
            return res

    def _gen_weldobj(self, arr):
        '''
        Generating a new WeldArray from a given arr.
        @arr: weldarray or ndarray.
            - weldarray: Just update the weldobject with the context from the
              weldarray.
            - ndarray: Add the given array to the context of the weldobject.
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

    def _process_ufunc_inputs(self, input_args, outputs):
        '''
        Helper function for __array_ufunc__ that deals with a lot of the
        annoying checks.
        @input_args: args to __array_ufunc__
        @outputs: specified outputs.

        @ret1: output: If one output is specified.
        @ret2: supported: If weld supports the given input/output format - if
        Weld doesn't then will just pass it on to numpy.
        '''
        if len(input_args) > 2:
            supported = False
            return None, supported

        arrays = all(isinstance(i, np.ndarray) for i in input_args)
        supported = True
        if arrays:
            supported = all(str(_inp.dtype) in SUPPORTED_DTYPES for _inp in input_args)
            if len(input_args) == 2 and input_args[0].dtype != input_args[1].dtype:
                supported = False
        else:
            # FIXME: add more scalar tests
            pass

        # check ouput.
        if outputs:
            # if the output is not the same weldarray as self, then let np deal
            # with it.
            if len(outputs) == 1 and outputs[0].weldobj.objectId == self.weldobj.objectId:
                output = outputs[0]
            else:
                # not sure about the use case with multiple outputs - let numpy
                # handle it.
                supported = False
                output = None
        else:
            output = None

        return output, supported

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Overwrites the action of weld supported functions and for others,
        pass on to numpy's implementation.

        FIXME: Not dealing with method at all. Check use cases.
        TODO: add input descriptions.
        '''
        input_args = [inp for inp in inputs]
        outputs = kwargs.pop('out', None)
        output, supported = self._process_ufunc_inputs(input_args, outputs)
        if output is None:
            # the new array will have no views parent/children
            self.view_parents = []
            self.view_children = []

        # check for supported ops.
        if ufunc.__name__ in UNARY_OPS and supported:
            assert(len(input_args) == 1)
            return self._unary_op(UNARY_OPS[ufunc.__name__], result=output)

        if ufunc.__name__ in BINARY_OPS and supported:
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

        # Relegating the work to numpy. If input arg is WeldArray, evaluate it,
        # and convert to ndarray before passing to super()
        for i, arg_ in enumerate(input_args):
            if isinstance(arg_, WeldArray):
                # Evaluate all the lazily stored computations first.
                input_args[i] = arg_._eval()
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
        # if possible, return weldarray.
        if str(result.dtype) in SUPPORTED_DTYPES:
            return WeldArray(result)
        else:
            return result

    def evaluate(self):
        '''
        User facing function - if he wants to explicitly evaluate all the
        registered ops.
        '''
        return WeldArray(self._eval())

    def _eval(self):
        '''
        @ret: ndarray after evaluating all the ops registered with the given
        WeldArray.

        Users should instead call evaluate which would return a WeldArray as
        expected. This returns an ndarray.

        Evalutes the expression based on weldobj.weld_code. If no new ops have
        been registered, then just returns the last ndarray.
        '''
        if self.name == self.weldobj.weld_code:
            # no new ops have been registered. Let's avoid creating
            # unneccessary new copies with weldobj.evaluate()
            arr = self.weldobj.context[self.name]
            return arr

        arr = self.weldobj.evaluate(WeldVec(self._weld_type),
                verbose=self.verbose)
        # Now that the evaluation is done - create new weldobject for self,
        # initalized from the given arr.
        self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        self.weldobj.weld_code = ''
        self.name = self.weldobj.update(arr, SUPPORTED_DTYPES[str(arr.dtype)])
        self.weldobj.weld_code = self.name

        return arr

    def _unary_op(self, unop, result=None):
        '''
        Create a new array, and updates weldobj.weld_code with the code for
        unop.
        @unop: str, weld IR for the given function.
        @result: output array.
        '''
        def _update_array_unary_op(res, unop):
            '''
            '''
            template = 'map({arr}, |z : {type}| {unop}(z))'
            code = template.format(arr = res.weldobj.weld_code,
                    type=res._weld_type.__str__(),
                    unop=unop)
            res.weldobj.weld_code = code

        if result is None:
            result = WeldArray(self)
        else:
            # if array had view / was a view - need to update the other
            # appropriate arrays too.
            for c in self.view_children:
                _update_array_unary_op(c, unop)

            # TODO: Updating parents.

        _update_array_unary_op(result, unop)
        return result

    def _scalar_binary_op(self, other, binop, result):
        '''
        Helper function for _binary_op.
        @other is a scalar (i32, i64, f32, f64).
        @result: None or a weldarray to store results in.
        '''
        if result is None:
            result = WeldArray(self)
        else:
            # FIXME: deal with updating view children/parents
            pass

        # FIXME: support other dtypes - not sure what to do for f64/or i64.
        if isinstance(other, np.float32) or isinstance(other, float):
            template = 'map({arr}, |z: {type}| z {binop} {other}f)'

        elif isinstance(other, np.int32) or isinstance(other, int):
            template = 'map({arr}, |z: {type}| z {binop} {other})'

        code = template.format(arr = result.weldobj.weld_code,
                    type = result._weld_type.__str__(),
                    binop = binop,
                    other = str(other))
        result.weldobj.weld_code = code

        return result

    def _binary_op(self, other, binop, result=None):
        '''
        @other: ndarray, weldarray or scalar.
        @binop: str, one of the weld supported binop.
        @result: output array. If it has been specified by the caller, then we
        don't have to allocate a new array.
        '''
        if not hasattr(other, "__len__"):
            return self._scalar_binary_op(other, binop, result)

        is_weld_array = False
        if isinstance(other, WeldArray):
            is_weld_array = True
        elif isinstance(other, np.ndarray):
            # turn it into WeldArray - for convenience, as this reduces to the
            # case of other being a WeldArray.
            # TODO: To optimize this, might want to update the weldobj without
            # converting other to WeldArray.
            other = WeldArray(other)
        else:
            raise ValueError('invalid array type for other in binaryop')

        # @arr{i}: latest array based on the ops registered on operand{i}. {{
        # is used for escaping { for format.
        template = 'map(zip({arr1},{arr2}), |z: {{{type1},{type2}}}| z.$0 {binop} z.$1)'

        if result is None:
            result = WeldArray(self)
        else:
            # if array had view / was a view - then update those too.
            for c in self.view_children:
                c.weldobj.update(other.weldobj)
                # for arr2, need to select the subset of other whose indices
                # matches c.
                start = (addr(c) - addr(self.weldobj.context[self.name])) / c.itemsize
                end = len(c)
                # FIXME: For Weld IR, mixing sugar syntax (map/zip) and low level constructs
                # like iter does not seem to be working.
                arr2_template = 'iter({arr},{start},{end},{stride})'
                arr2 = arr2_template.format(arr=other.weldobj.weld_code,
                                            start = str(start),
                                            end = str(end),
                                            stride = str(1))
                c.weldobj.weld_code = template.format(arr1=c.weldobj.weld_code,
                                                      arr2=arr2,
                                                      type1 = c._weld_type.__str__(),
                                                      type2 = other._weld_type.__str__(),
                                                      binop = binop)

        result.weldobj.update(other.weldobj)
        code = template.format(arr1 = result.weldobj.weld_code,
                arr2 = other.weldobj.weld_code,
                type1 = result._weld_type.__str__(),
                type2 = other._weld_type.__str__(),
                binop = binop)
        result.weldobj.weld_code = code

        if not is_weld_array:
            # Evaluate all the lazily stored ops, as the caller probably
            # expects to get back a ndarray - which would have correct values.
            result = WeldArray(result._eval())

        return result

# TODO: This should finally go into weldnumpy - along with other functions
# which are equivalents of np.zeros etc. - need to figure out the exact
# imports etc.
def array(arr, *args, **kwargs):
    '''
    Wrapper around WeldArray - first create np.array and then convert to
    WeldArray.
    '''
    return WeldArray(np.array(arr, *args, **kwargs))
