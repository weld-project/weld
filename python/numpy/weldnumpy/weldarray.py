from weld.weldobject import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
from weldnumpy import *

class weldarray(np.ndarray):
    '''
    A new weldarray can be created in three ways:
        - Explicit constructor. This is dealt with in __new__
        - Generating view of weldarray (same as ndarray)
        - fancy indexing (same as ndarray)

    The second and third case do not pass through __new__.  Instead, numpy guarantees that after
    initializing an array, in all three methods, numpy will call __array_finalize__. But we are
    able to deal with the 2nd and 3rd case in __getitem__ so there does not seem to be any need to
    use __array_finalize (besides __array_finalize__ also adds a function call to the creation of a
    new array, which adds to the overhead compared to numpy for initializing arrays)
    '''
    def __new__(cls, input_array, verbose=True, *args, **kwargs):
        '''
        @input_array: original ndarray from which the new array is derived.
        '''
        # TODO: Add support for lists / ints.
        assert isinstance(input_array, np.ndarray), 'only support ndarrays'
        assert str(input_array.dtype) in SUPPORTED_DTYPES

        # sharing memory with the ndarray/weldarry
        obj = np.asarray(input_array).view(cls)
        obj._gen_weldobj(input_array)

        obj._weld_type = SUPPORTED_DTYPES[str(input_array.dtype)]
        obj._verbose = verbose

        # Views. For a base_array, this would always be None.
        obj._weldarray_view = None

        return obj

    def __array_finalize__(self, obj):
        '''
        array finalize will be called whenever a subclass of ndarray (e.g., weldarray) is created.
        This can happen in many situations where it does not go through __new__, e.g.:
            - np.float32(arr)
            - arr.T
        Thus, we want to generate the generic variables needed for the weldarray here.

        @self: weldarray, current weldarray object being created.
        @obj:  weldarray or ndarray, depending on whether the object is being created based on a
        ndarray or a weldarray. If obj is a weldarray, then __new__ was not called for self, so we
        should update the appropriate fields.

        TODO: need to consider edge cases with views etc.
        '''
        if isinstance(obj, weldarray):
            # self was not generated through a call to __new__ so we should update self's
            # properties
            self.name = obj.name
            # TODO: Or maybe just set them equal to each other?
            self._gen_weldobj(obj)
            self._verbose = obj._verbose
            self._weldarray_view = obj._weldarray_view
            self._weld_type = obj._weld_type

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
        TODO: Multidimensional support. Will make it a lot more complicated.
        arr[...] type of indexing.
        Deal with the following three scenarios:
            - idx is a scalar
            - idx is an array (fancy indicing)
                - Steps: Both the cases above are dealt in the same way. They are simpler because
                  in numpy they aren't supposed to share memory with the parent arrays.
                    1. _eval self to get the latest values in the array.
                    2. Call np.ndarray's __getitem__ method on self.
                    3. Convert to weldarray and return.

            - idx is a slice (e.g., 3:5)
                - In this scenario, if the memory is being shared with the parent then we do not
                  evaluate the ops stored so far in the base array (the original array which the
                  view is a subset of). Instead future in place ops on the
                  view are just added to the base array.
                - Steps:
                    1. Call np.ndarray's __getitem__ method on self.
                    2. Convert to weldarray.
                    3. Create the arr._weldarray_view class for the view which stores pointers to
                    base_array, parent_array, and start/end/strides/idx values.
        '''
        # Need to cast it as ndarray view before calling ndarray's __getitem__ implementation.
        ret = self.view(np.ndarray).__getitem__(idx)
        if isinstance(idx, slice):
            # TODO: The way we treat views now, views don't need their own weldobj - just the
            # weldview object. Could be a minor optimization.
            ret = weldarray(ret, verbose=self._verbose)
            # check if ret really is a view of arr. If not, then don't need to make changes to
            # ret's weldarray_view etc.
            if is_view_child(ret.view(np.ndarray), self.view(np.ndarray)):
                if self._weldarray_view is None:
                    base_array = self
                    par_start = 0
                else:
                    base_array = self._weldarray_view.base_array
                    par_start = self._weldarray_view.start

                # start / end is relative to the base_array because all future in place updates to
                # the view would be made on the relevant indices on the base array.
                start = par_start + idx.start
                end = par_start + idx.stop
                # ret is a view, initialize its weldview.
                ret._weldarray_view = weldarray_view(base_array, self, start, end, idx)

            return ret

        elif isinstance(idx, np.ndarray) or isinstance(idx, list):
            # FIXME: This could be a view if list is contig numbers? Test + update.
            arr = self._eval()
            # call ndarray's implementation on it.
            ret = arr.__getitem__(idx)
            return weldarray(ret, verbose=self._verbose)
        elif isinstance(idx, int):
            arr = self._eval()
            ret = arr.__getitem__(idx)
            # return the scalar.
            return ret
        else:
            assert False, 'idx type not supported'

    def __setitem__(self, idx, val):
        '''
        Cases:
            1. arr[idx] = num
            2. arr[idx] += num
                - Both these are the same as np handles calculating the inplace increment and
                  num is the latest value in the array.
            3. idx can be:
                - slice
                - ndarray
                - int

        When self is a view, update parent instead.
        TODO: This is work in progress, although it does seem to be mostly functionally correct for now.
        '''
        def _update_single_entry(arr, index, val):
            '''
            @start: index to update.
            @val: new val for index.
            called from idx = int, or list.
            '''
            # update just one element
            suffix = DTYPE_SUFFIXES[self._weld_type.__str__()]
            update_str = str(val) + suffix
            arr._update_range(index, index+1, update_str)

        if isinstance(idx, slice):
            if idx.step is None: step = 1
            else: step = idx.step
            # FIXME: hacky - need to check exact mechanisms b/w getitem and setitem calls further.
            # + it seems like numpy calls getitem and evaluates the value, and then sets it to the
            # correct value here (?) - this seems like a waste.
            if self._weldarray_view:
                for i, e in enumerate(range(idx.start, idx.stop, step)):
                    _update_single_entry(self._weldarray_view.base_array, e, val[i])
            else:
                # FIXME: In general, this sucks for performance - instead add the list as an array to the
                # weldobject context and use lookup on it for the weld IR code.
                # just update it for each element in the list
                for i, e in enumerate(range(idx.start, idx.stop, step)):
                    _update_single_entry(self, e, val[i])

        elif isinstance(idx, np.ndarray) or isinstance(idx, list):
            # just update it for each element in the list
            for i, e in enumerate(idx):
                _update_single_entry(self, e, val[i])

        elif isinstance(idx, int):
            _update_single_entry(self, idx, val)
        else:
            assert False, 'idx type not supported'

    def _gen_weldobj(self, arr):
        '''
        Generating a new weldarray from a given arr for self.
        @arr: weldarray or ndarray.
            - weldarray: Just update the weldobject with the context from the
              weldarray.
            - ndarray: Add the given array to the context of the weldobject.
        Sets self.name and self.weldobj.
        '''
        self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
        if isinstance(arr, weldarray):
            self.weldobj.update(arr.weldobj)
            self.weldobj.weld_code = arr.weldobj.weld_code
            self.name = arr.name
        else:
            # general case for arr being numpy scalar or ndarray
            # weldobj returns the name bound to the given array. That is also
            # the array that future ops will act on, so set weld_code to it.
            self.name = self.weldobj.weld_code = self.weldobj.update(arr,
                    SUPPORTED_DTYPES[str(arr.dtype)])

    def _process_ufunc_inputs(self, input_args, outputs):
        '''
        Helper function for __array_ufunc__ that deals with the input/output checks and determines
        if this should be relegated to numpy's implementation or not.
        @input_args: args to __array_ufunc__
        @outputs: specified outputs.

        @ret: Bool, If weld supports the given input/output format or not - if weld doesn't then
        will just pass it on to numpy.
        '''
        if len(input_args) > 2:
            return False

        arrays = []
        scalars = []
        for i in input_args:
            if isinstance(i, np.ndarray):
                if not str(i.dtype) in SUPPORTED_DTYPES:
                    return False
                if len(i) == 0:
                    return False
                arrays.append(i)
            elif isinstance(i, list):
                return False
            else:
                scalars.append(i)

        if len(arrays) == 2 and arrays[0].dtype != arrays[1].dtype:
            return False

        # handle all scalar based tests here - later will just assume that scalar type is correct,
        # and use the suffix based on the weldarray's type.
        # TODO: add more scalar tests for python int/float: floats -> f32, f64, or ints -> i32, i64 need to
        # match).
        elif len(arrays) == 1 and len(scalars) == 1:
            # need to test for bool before int because it True would be instance of int as well.
            if isinstance(scalars[0], bool):
                return False
            elif isinstance(scalars[0], float):
                pass
            elif isinstance(scalars[0], int):
                pass
            # assuming its np.float32 etc.
            elif not str(scalars[0].dtype) in SUPPORTED_DTYPES:
                return False
            else:
                weld_type = SUPPORTED_DTYPES[str(scalars[0].dtype)]
                if weld_type != arrays[0]._weld_type:
                    return False
        # check ouput.
        if outputs:
            # if the output is not weldarray, then let np deal with it.
            if not (len(outputs) == 1 and isinstance(outputs[0], weldarray)):
                return False

        return True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Overwrites the action of weld supported functions and for others,
        pass on to numpy's implementation.
        @ufunc: np op, like np.add etc.
        TODO: support other type of methods?
        @method: call, __reduce__,
        '''
        input_args = [inp for inp in inputs]
        outputs = kwargs.pop('out', None)
        supported = self._process_ufunc_inputs(input_args, outputs)
        output = None
        if supported and method == '__call__':
            output = self._handle_call(ufunc, input_args, outputs)
        elif supported and method == 'reduce':
            output = self._handle_reduce(ufunc, input_args, outputs)

        if output is not None:
            return output

        return self._handle_numpy(ufunc, method, input_args, outputs, kwargs)

    def _handle_numpy(self, ufunc, method, input_args, outputs, kwargs):
        '''
        relegate responsibility of executing ufunc to numpy.
        '''
        # Relegating the work to numpy. If input arg is weldarray, evaluate it,
        # and convert to ndarray before passing to super()
        for i, arg_ in enumerate(input_args):
            if isinstance(arg_, weldarray):
                # Evaluate all the lazily stored computations first.
                input_args[i] = arg_._eval()

        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, weldarray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        result = super(weldarray, self).__array_ufunc__(ufunc, method, *input_args, **kwargs)

        # if possible, return weldarray.
        if str(result.dtype) in SUPPORTED_DTYPES and isinstance(result, np.ndarray):
            return weldarray(result, verbose=self._verbose)
        else:
            return result

    def _handle_call(self, ufunc, input_args, outputs):
        '''
        TODO: add description
        '''
        # if output is not None, choose the first one - since we don't support any ops with
        # multiple outputs.
        if outputs: output = outputs[0]
        else: output = None

        # check for supported ops.
        if ufunc.__name__ in UNARY_OPS:
            assert(len(input_args) == 1)
            return self._unary_op(UNARY_OPS[ufunc.__name__], result=output)

        if ufunc.__name__ in BINARY_OPS:
            # weldarray can be first or second arg.
            if isinstance(input_args[0], weldarray):
                # first arg is weldarray, must be self
                assert input_args[0].name == self.name
                other_arg = input_args[1]
            else:
                other_arg = input_args[0]
                assert input_args[1].name == self.name

            return self._binary_op(other_arg, BINARY_OPS[ufunc.__name__], result=output)

    def _handle_reduce(self, ufunc, input_args, outputs):
        '''
        TODO: describe.
        TODO: For multi-dimensional case, might need extra work here if the reduction is being
        performed only along a certain axis.
        We force evaluation at the end of the reduction - weird errors if we don't. This seems
        safer anyway.
        np supports reduce only for binary ops.
        '''
        # input_args[0] must be self so it can be ignored.
        assert len(input_args) == 1
        if outputs: output = outputs[0]
        else: output = None
        if ufunc.__name__ in BINARY_OPS:
            return self._reduce_op(BINARY_OPS[ufunc.__name__], result=output)

    def _reduce_op(self, op, result=None):
        '''
        TODO: support for multidimensional arrays.
        '''
        template = 'result(for({arr},merger[{type}, {op}], |b, i, e| merge(b, e)))'
        self.weldobj.weld_code = template.format(arr = self.weldobj.weld_code,
                                                 type = self._weld_type.__str__(),
                                                 op  = op)
        return self._eval(restype = self._weld_type)

    def evaluate(self):
        '''
        User facing function - if he wants to explicitly evaluate all the
        registered ops.
        '''
        return weldarray(self._eval(), verbose=self._verbose)

    def _eval(self, restype=None):
        '''
        @ret: ndarray after evaluating all the ops registered with the given weldarray.
        @restype: type of the result. Usually, it will be a WeldVec, but if called from reduction,
        it would be a scalar.
        Evalutes the expression based on weldobj.weld_code. If no new ops have been registered,
        then just returns the last ndarray.
        If self is a view, then evaluates the parent array, and returns the aprropriate index from
        the result.
        Internal call - used at various points to implicitly evaluate self. Users should instead
        call evaluate which would return a weldarray as expected. This returns an ndarray.
        '''
        # This check has to happen before the caching - as the weldobj/code for views is never updated.
        if self._weldarray_view:
            # _eval parent and return appropriate idx.
            # TODO: Clearly more efficient to _eval the base array as the parent would eventually have to do
            # it - but this makes it more convenient to deal with different indexing strategies.
            arr = self._weldarray_view.parent._eval()
            return arr[self._weldarray_view.idx]

        # Caching
        if self.name == self.weldobj.weld_code:
            # No new ops have been registered. Avoid creating unneccessary new copies with
            # weldobj.evaluate()
            return self.weldobj.context[self.name]

        if restype is None:
            # use default type for all weldarray operations
            restype = WeldVec(self._weld_type)
        arr = self.weldobj.evaluate(restype, verbose=self._verbose)
        # Now that the evaluation is done - create new weldobject for self,
        # initalized from the returned arr.
        self._gen_weldobj(arr)
        return arr

    def _get_result(self):
        '''
        Creating a new result weldarray from self. If self is view into a weldarray, then evaluate
        the parent first as self would not be storing the ops that have been registered to it (only
        base_array would store those).
        '''
        if self._weldarray_view:
            idx = self._weldarray_view.idx
            result = weldarray(self._weldarray_view.parent._eval()[idx], verbose=self._verbose)
        else:
            result = weldarray(self, verbose=self._verbose)
        return result

    def _unary_op(self, unop, result=None):
        '''
        @unop: str, weld IR for the given function.
        @result: output array.

        Create a new array, and updates weldobj.weld_code with the code for
        unop.
        '''
        def _update_array_unary_op(res, unop):
            '''
            @res: weldarray to be updated.
            @unop: str, operator applied to res.
            '''
            template = 'map({arr}, |z : {type}| {unop}(z))'
            code = template.format(arr  = res.weldobj.weld_code,
                                   type = res._weld_type.__str__(),
                                   unop = unop)
            res.weldobj.weld_code = code

        if result is None:
            result = self._get_result()
        else:
            # in place op. If is a view, just update base array and return.
            if result._weldarray_view:
                v = result._weldarray_view
                update_str = '{unop}(e)'.format(unop=unop)
                v.base_array._update_range(v.start, v.end, update_str)
                return result

        # back to updating result array
        _update_array_unary_op(result, unop)
        return result

    def _scalar_binary_op(self, other, binop, result):
        '''
        Helper function for _binary_op.
        @other, scalar values (i32, i64, f32, f64).
        @result: weldarray to store results in.
        '''
        template = 'map({arr}, |z: {type}| z {binop} {other}{suffix})'
        weld_type = result._weld_type.__str__()
        result.weldobj.weld_code = template.format(arr = result.weldobj.weld_code,
                                                  type =  weld_type,
                                                  binop = binop,
                                                  other = str(other),
                                                  suffix = DTYPE_SUFFIXES[weld_type])
        return result

    def _update_range(self, start, end, update_str, strides=1):
        '''
        @start, end: define which values of the view needs to be updated - for a child, it
        would be all values, and for parent it wouldn't be.
        @update_str: code to be executed in the if block to update the variable, 'e', in the given
        range.
        '''
        views_template = "result(for({arr1}, appender,|b,i,e| if \
        (i >= {start}L & i < {end}L,merge(b,{update_str}),merge(b,e))))"

        # all values of child will be updated. so start = 0, end = len(c)
        self.weldobj.weld_code = views_template.format(arr1 = self.weldobj.weld_code,
                                                      start = start,
                                                      end   = end,
                                                      update_str = update_str)

    def _update_views_binary(self, result, other, binop):
        '''
        @result: weldarray that is being updated
        @other: weldarray or scalar. (result binop other)
        @binop: str, operation to perform.

        FIXME: the common indexing pattern for parent/child in _update_view might be too expensive
        (uses if statements (unneccessary checks when updating child) and wouldn't be ideal to
        update a large parent).
        '''
        update_str_template = '{e2}{binop}e'
        v = result._weldarray_view
        if isinstance(other, weldarray):
            lookup_ind = 'i-{st}'.format(st=v.start)
            # update the base array to include the context from other
            v.base_array.weldobj.update(other.weldobj)
            e2 = 'lookup({arr2},{i}L)'.format(arr2 = other.weldobj.weld_code, i = lookup_ind)
        else:
            # other is just a scalar.
            e2 = str(other) + DTYPE_SUFFIXES[result._weld_type.__str__()]

        update_str = update_str_template.format(e2 = e2, binop=binop)
        v.base_array._update_range(v.start, v.end, update_str)

    def _binary_op(self, other, binop, result=None):
        '''
        @other: ndarray, weldarray or scalar.
        @binop: str, one of the weld supported binop.
        @result: output array. If it has been specified by the caller, then
        don't allocate new array.
        '''
        if isinstance(other, weldarray):
            if other._weldarray_view:
                idx = other._weldarray_view.idx
                other = weldarray(other._weldarray_view.parent._eval()[idx],verbose=self._verbose)

        elif isinstance(other, np.ndarray):
            # TODO: could optimize this by skipping conversion to weldarray.
            # turn it into weldarray - for convenience, as this reduces to the
            # case of other being a weldarray.
            other = weldarray(other)

        # Dealing with views. This is the same for other being scalar or weldarray.
        if result is None:
            # New array being created, so don't deal with views.
            result = self._get_result()
        else:
            # for a view, we only update the parent array.
            if result._weldarray_view:
                self._update_views_binary(result, other, binop)
                return result

        # scalars (i32, i64, f32, f64...)
        if not hasattr(other, "__len__"):
            return self._scalar_binary_op(other, binop, result)

        result.weldobj.update(other.weldobj)
        # @arr{i}: latest array based on the ops registered on operand{i}. {{
        # is used for escaping { for format.
        template = 'map(zip({arr1},{arr2}), |z: {{{type1},{type2}}}| z.$0 {binop} z.$1)'
        result.weldobj.weld_code = template.format(arr1 = result.weldobj.weld_code,
                                                  arr2  = other.weldobj.weld_code,
                                                  type1 = result._weld_type.__str__(),
                                                  type2 = other._weld_type.__str__(),
                                                  binop = binop)
        return result
