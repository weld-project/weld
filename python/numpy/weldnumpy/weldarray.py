from weld.weldobject import WeldObject
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
from weldnumpy import *
import weldnumpy as wn
import numpy as np

class weldarray(np.ndarray):
    '''
    A new weldarray can be created in three ways:
        - Explicit constructor. This is dealt with in __new__
        - Generating view of weldarray (same as ndarray)
        - fancy indexing (same as ndarray)

    The second and third case do not pass through __new__.  Instead, numpy
    guarantees that after initializing an array, in all three methods, numpy
    will call __array_finalize__. But we are able to deal with the 2nd and 3rd
    case in __getitem__ so there does not seem to be any need to use
    __array_finalize (besides __array_finalize__ also adds a function call to
    the creation of a new array, which adds to the overhead compared to numpy
    for initializing arrays)
    '''
    def __new__(cls, input_array, verbose=False, new_weldobj = True, *args, **kwargs):
        '''
        @input_array: original ndarray from which the new array is derived.
        '''
        # TODO: Add support for lists / ints.
        assert isinstance(input_array, np.ndarray), 'only support ndarrays'
        assert str(input_array.dtype) in SUPPORTED_DTYPES
        # sharing memory with the ndarray/weldarray. Might be interesting to
        # further think whether we should use np.asanyarray - which would pass
        # through the subclases, thus create a new weldarray / and weldobject etc.
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
            # in case of ndarray.T it doesn't seem like self.shape or obj.shape
            # is correct at this stage
            self._real_shape = self.shape

        # since an array can be created without explicitly going through
        # __new__, we need to do these intializations here.
        if hasattr(obj, '_num_registered_ops'):
            self._num_registered_ops = obj._num_registered_ops
        else:
            self._num_registered_ops = 0

        if hasattr(obj, '_real_shape'):
            self._real_shape = obj._real_shape
        else:
            self._real_shape = self.shape

    def transpose(self, *args, **kwargs):
        '''
        numpy's transpose checks if the particular array subclass being
        transposed has an implementation for transpose -- and calls it if the
        subclass does.
        high level pseudo-code:
            - force evaluation
            - offload to numpy to perform reshape
            - return a new weldarray
        TODO: not clear if we need to always force evaluation - will need to
        test more extensively.
        '''
        # since we can't modify tuples, need to make it a list to change the
        # first arg to a list
        args_list = list(args)
        new_arr = self._eval()
        if (len(args_list) >= 1):
            args_list[0] = new_arr.view(np.ndarray)
        else:
            args_list.append(new_arr.view(np.ndarray))
        args = tuple(args_list)
        # let numpy handle transposing.
        transposed = np.transpose(*args, **kwargs)
        real_shape = transposed.shape
        # create the weldarray view and initialize relevant fields of weldarray
        if isinstance(self, weldarray):
            transposed = self._gen_weldview(transposed)
            transposed._real_shape = real_shape
        return transposed

    def reshape(self, *args, **kwargs):
        '''
        numpy's reshape checks if the particular array subclass being reshaped
        has an implementation for reshape -- and calls it if the subclass does.
        high level pseudo-code:
            - force evaluation
            - offload to numpy to perform reshape
            - return a new weldarray
        '''
        # force evaluation, just in case. Should not be strictly neccessary but
        # then there would be edge cases to think about.
        new_arr = self._eval()
        new_args = list(args)
        if isinstance(args[0], weldarray):
            # case when np.reshape(w, shape) used
            new_args[0] = new_arr
        else:
            # this is the case when w.reshape(shape) used
            new_args.insert(0, new_arr)
        new_args = tuple(new_args)
        reshaped_arr = np.reshape(*new_args, **kwargs)
        return weldarray(reshaped_arr)

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

    def _gen_weldview(self, new_arr, idx=None):
        '''
        TODO: can generalize this to the 1d cases too or not?
        @new_arr: new view of self.
        @idx: the idx used to create the view. Not really needed - because
        there are cases when this function is being used from outside
        __getitem__, e.g., for views created by broadcasting, or transposes.
        FIXME: Want to do this so that we can modify self's weldarray view
        directly?
        @returns: new_arr with the weldview being generated.
        '''
        new_arr = weldarray(new_arr, verbose=self._verbose)
        # FIXME: Do stuff with views being grandchildren here.
        # now let's create the weldarray view with the starts/stops/steps
        if self._weldarray_view is None:
            base_array = self
            par_start = 0
        else:
            # assert False, 'nested views, need to fix this still'
            base_array = self._weldarray_view.base_array
            par_start = self._weldarray_view.start

        # FIXME: cleaner way to calculate start?
        start = (addr(new_arr) - addr(base_array)) / self.itemsize
        start += par_start
        # TODO: temporarily needed.
        end = 1
        shape = new_arr.shape
        strides = new_arr.strides
        for s in shape:
            end = end*s
        end += start

        # new_arr is a view, so initialize its weldview.
        new_arr._weldarray_view = weldarray_view(base_array, self, start, end,
                idx, shape=shape, strides=strides)
        return new_arr

    def __getitem__(self, idx):
        '''
        Deal with the following scenarios:
            - idx is a scalar
            - idx is an array (fancy indicing)
                - Steps: Both the cases above are dealt in the same way. They
                  are simpler because in numpy they aren't supposed to share
                  memory with the parent arrays.
                    1. _eval self to get the latest values in the array.
                    2. Call np.ndarray's __getitem__ method on self.
                    3. Convert to weldarray and return.

            - idx is a slice (e.g., 3:5)
                - In this scenario, if the memory is being shared with the
                  parent then we do not evaluate the ops stored so far in the
                  base array (the original array which the view is a subset
                  of). Instead future in place ops on the view are just added
                  to the base array.
                - Steps:
                    1. Call np.ndarray's __getitem__ method on self.
                    2. Convert to weldarray.
                    3. Create the arr._weldarray_view class for the view which
                    stores pointers to base_array, parent_array, and
                    start/end/strides/idx values.

            - idx is a tuple: Used for multi-dimensional arrays.
                - TODO: explain stuff + similarities/differences with slices.
        '''
        # Need to cast it as ndarray view before calling ndarray's __getitem__
        # implementation.
        if isinstance(idx, slice):
            # TODO: The way we treat views now, views don't need their own
            # weldobj - just the weldview object. Could be a minor
            # optimization.
            ret = self.view(np.ndarray).__getitem__(idx)
            ret = weldarray(ret, verbose=self._verbose)
            # check if ret really is a view of arr. If not, then don't need to
            # make changes to ret's weldarray_view etc.
            if is_view_child(ret.view(np.ndarray), self.view(np.ndarray)):
                if self._weldarray_view is None:
                    base_array = self
                    par_start = 0
                else:
                    base_array = self._weldarray_view.base_array
                    par_start = self._weldarray_view.start

                # start / end is relative to the base_array because all future
                # in place updates to the view would be made on the relevant
                # indices on the base array.
                start = par_start + idx.start
                end = par_start + idx.stop
                # ret is a view, initialize its weldview.
                strides = ret.strides
                ret._weldarray_view = weldarray_view(base_array, self, start,
                        end, idx, strides=strides)

            return ret

        elif isinstance(idx, tuple):
            # FIXME: This seems like it WOULD FAIL w/o an evaluate?
            ret = self.view(np.ndarray).__getitem__(idx)
            if isinstance(ret, np.ndarray):
                return self._gen_weldview(ret, idx)
            else:
                # could have been just a float, so return it without converting
                # to weldarray
                return ret

        elif isinstance(idx, np.ndarray) or isinstance(idx, list):
            # FIXME: This could be a view if list is contig numbers? Test +
            # update.
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
        idx can be:
            - slice
            - ndarray
            - int
        In general, we follow the strategy of offloading everything to NumPy
        after evaluating the stored operations.
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

        # In general, we just want to offload setitems until we can support in
        # place ops in NumPy.
        if wn.offload_setitem:
            # We want to evaluate any stored ops first. This is crucial in all
            # situations where we offload stuff to NumPy. Then we can update
            # the arrays in place using NumPy - knowing that future ops that
            # are registered on this weldarray will utilize the new memory
            # location - where these changed would have occurred.
            if self._weldarray_view is not None:
                # in general, we store all ops in base arrays, so we evaluate
                # that first, and then get the latest version of the view.
                view_idx = self._weldarray_view.idx
                view_strides = self._weldarray_view.strides
                base_array = self._weldarray_view.base_array._eval().view(np.ndarray)
                latest_arr = base_array[view_idx]
                latest_arr.strides = view_strides
            else:
                latest_arr = self._eval()
            # Now, the idx could have been of different kinds
            if isinstance(idx, slice):
                # handle index first
                if idx.step is None: step = 1
                else: step = idx.step
                if idx.stop is None: stop = len(self)
                # this case weirdly happens some times...
                elif idx.stop > len(self): stop = len(self)
                else: stop = idx.stop
                if idx.start is None: start = 0
                else: start = idx.start

                # surprisingly enough, doing latest_arr_i.__setitem__(idx, val)
                # fails here for the case when we do a[..] += 5. It appears
                # that += is implemented as:
                #       >>> a.__setitem__(i, a.__getitem__(i).__iadd__(x))
                # and so val already has the correct value -- but since the add
                # was done on a weldarray, it was recorded in it, and the above
                # evaluation already causes latest_arr to have the value it
                # should.
                # In the second case, with a[..] = 5, things work fine with
                # setitem as well.
                # Somehow, switching the setitem call to explicitly looping
                # over the array and setting each element to val (which has
                # already been calculated to the correct value in both cases)
                # seems to work just fine.
                # TODO: look further into NumPy's setitem implementation to see
                # what is going wrong.
                for val_i, latest_arr_i  in enumerate(range(start, stop, step)):
                    if hasattr(val, '__len__'):
                        latest_arr[latest_arr_i] = val[val_i]
                    else:
                        # also a single float can be assigned to multiple values
                        latest_arr[latest_arr_i] = val

            elif isinstance(idx, np.ndarray) or isinstance(idx, list) \
            or isinstance(idx, tuple):
                # each element of the idx must be the index into our array
                for val_i, arr_i in enumerate(idx):
                    if hasattr(val, '__len__'):
                        latest_arr[arr_i] = val[val_i]
                    else:
                        # must be a float
                        latest_arr[arr_i] = val

            elif isinstance(idx, int):
                latest_arr[idx] = val
            else:
                assert False, 'idx type {} not supported'.format(type(idx))
        else:
            # in general, this seems to work, but it is just horribly slow and
            # MUST be avoided until we can figure out a way to implement
            # _update_single_entry efficiently in weld using in place ops or
            # so.
            if isinstance(idx, slice):
                if idx.step is None: step = 1
                else: step = idx.step
                # FIXME: hacky - need to check exact mechanisms b/w getitem and
                # setitem calls further.  + it seems like numpy calls getitem
                # and evaluates the value, and then sets it to the correct
                # value here (?) - this seems like a waste.
                if self._weldarray_view:
                    for i, e in enumerate(range(idx.start, idx.stop, step)):
                        # the index should be appropriate for the base array
                        e += self._weldarray_view.start
                        _update_single_entry(self._weldarray_view.base_array, e, val[i])
                else:
                    # FIXME: In general, this sucks for performance - instead
                    # add the list as an array to the weldobject context and
                    # use lookup on it for the weld IR code.  just update it
                    # for each element in the list
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

    def _process_ufunc_inputs(self, input_args, outputs, kwargs):
        '''
        Helper function for __array_ufunc__ that deals with the input/output
        checks and determines if this should be relegated to numpy's
        implementation or not.
        @input_args: args to __array_ufunc__
        @outputs: specified outputs.

        @ret: Bool, If weld supports the given input/output format or not - if
        weld doesn't then will just pass it on to numpy.
        '''
        if len(input_args) > 2:
            if self._verbose:
                print('WARNING: Length of input args > 2. Will be offloaded to numpy')
            return False

        arrays = []
        scalars = []
        shapes = []
        for i in input_args:
            # FIXME: This enforces that if ndarray op weldarray, then we force
            # evaluation with weldarray. Usually, this does not seem required,
            # but somehow, in cases with non-contiguous arrays, it seems to
            # fail otherwise. Might just be a bug in weldnumpy, will need to
            # explore further.
            if isinstance(i, np.ndarray) and not isinstance(i, weldarray):
                return False

            if isinstance(i, np.ndarray):
                if not str(i.dtype) in SUPPORTED_DTYPES:
                    if self._verbose: print('WARNING {} not in supported dtypes. Will be offloaded to \
                            numpy'.format(str(i.dtype)))
                    return False
                if len(i) == 0:
                    if self._verbose: print('WARNING: length 0 array. Will be offloaded to numpy')
                    return False
                if isinstance(i, weldarray):
                    shapes.append(i._real_shape)
                else:
                    shapes.append(i.shape)
                if outputs and not (outputs[0] is None or i.flags.contiguous):
                    # Inplace ops for non-contiguous inputs not supported
                    return False

                arrays.append(i)
            elif isinstance(i, list):
                if self._verbose: print('WARNING: input is a list. Will be offloaded to numpy')
                return False
            else:
                scalars.append(i)

        if len(arrays) == 2 and arrays[0].dtype != arrays[1].dtype:
            if self._verbose: print('WARNING: array dtypes do not match. Will be offloaded to numpy')
            return False
        elif len(arrays) == 2 and shapes[0] != shapes[1]:
            if self._verbose: print('WARNING: array shapes dont match. Will probably need to be broadcast')
            return False

        # handle all scalar based tests here - later will just assume that
        # scalar type is correct, and use the suffix based on the weldarray's
        # type.
        # TODO: add more scalar tests for python int/float: floats -> f32, f64,
        # or ints -> i32, i64 need to match).
        elif len(arrays) == 1 and len(scalars) == 1:
            # need to test for bool before int because it True would be
            # instance of int as well.
            if isinstance(scalars[0], bool):
                if self._verbose:
                    print('WARNING: scalar input is boolean. Will be offloaded to numpy')
                return False
            elif isinstance(scalars[0], float):
                if not (arrays[0].dtype == np.float64 or
                        arrays[0].dtype == np.float32):
                    return False
            elif isinstance(scalars[0], int):
                if not (arrays[0].dtype == np.int64 or
                        arrays[0].dtype == np.int32):
                    return False
            # assuming its np.float32 etc.
            elif not str(scalars[0].dtype) in SUPPORTED_DTYPES:
                if self._verbose:
                    print('WARNING: scalar dtype not in supported dtypes. Will be offloaded to numpy')
                return False
            else:
                scalar_weld_type = SUPPORTED_DTYPES[str(scalars[0].dtype)]
                array_weld_type = arrays[0]._weld_type
                if scalar_weld_type != array_weld_type:
                    if self._verbose:
                        print('WARNING: scalar weld type {} != array weld type {}'.format
                                (scalar_weld_type, array_weld_type))
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
        @ufunc: numpy op, like np.add etc.
        TODO: support other type of methods?
        @method: call, __reduce__,
        '''
        input_args = [inp for inp in inputs]
        outputs = kwargs.pop('out', None)
        supported = self._process_ufunc_inputs(input_args, outputs, kwargs)
        output = None
        if self._num_registered_ops > 100:
            # force evaluation internally.
            self._eval()

        if supported and method == '__call__':
            output = self._handle_call(ufunc, input_args, outputs, kwargs)
        elif supported and method == 'reduce':
            output = self._handle_reduce(ufunc, input_args, outputs, kwargs)

        if output is not None:
            if self._verbose: print('ufunc was supported ', ufunc)
            output._num_registered_ops += 1
            return output

        return self._handle_numpy(ufunc, method, input_args, outputs, kwargs)

    def _handle_numpy(self, ufunc, method, input_args, outputs, kwargs):
        '''
        relegate responsibility of executing ufunc to numpy.
        '''
        if self._verbose: print('WARNING: ufunc being offloaded to numpy', ufunc)
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

        result = super(weldarray, self).__array_ufunc__(ufunc, method,
                *input_args, **kwargs)

        # if possible, return weldarray.
        if str(result.dtype) in SUPPORTED_DTYPES and isinstance(result,
                np.ndarray):
            return weldarray(result, verbose=self._verbose)
        else:
            return result

    def _handle_call(self, ufunc, input_args, outputs, kwargs):
        '''
        TODO: add description
        '''
        # if output is not None, choose the first one - since we don't support
        # any ops with multiple outputs.
        if outputs: output = outputs[0]
        else: output = None

        # check for supported ops.
        if ufunc.__name__ in wn.UNARY_OPS:
            assert(len(input_args) == 1)
            if self._verbose: print('supported op: ', ufunc.__name__)
            return self._unary_op(wn.UNARY_OPS[ufunc.__name__], result=output)
        elif ufunc.__name__ in wn.BINARY_OPS:
            # weldarray can be first or second arg.
            if self._verbose: print('supported op: ', ufunc.__name__)
            return self._binary_op(input_args[0], input_args[1],
                    wn.BINARY_OPS[ufunc.__name__], result=output)

        # elif ufunc.__name__ == 'square' or ufunc.__name__ == 'power':
            # if ufunc.__name__ == 'square':
                # # power arg is implied
                # power = 2
            # else:
                # power = input_args[1]

            # if self._verbose: print('supported op: ', ufunc.__name__)
            # return self._power_op(power, result=output)

        # FIXME: Not doing this because numpy returns Boolean array -- and if we do that, then we can't
        # multiply it with f64 arrays in weld because of type mismatch.
        # elif ufunc.__name__ in wn.CMP_OPS:
            # return self._cmp_op(input_args[1], ufunc.__name__, result=output)

        return None

    def _cmp_op(self, input2, op, result=None):
        '''
        Only really works for floats right now.
        '''
        assert result is None, 'TODO: support this'
        if result is None:
            result = self._get_result()
        op = wn.CMP_OPS[op]
        template = ('result(for({arr}, appender,|b,i,e| if (e {op} {input2}{suffix},'
                'merge(b,1.0{suffix}),merge(b,0.0{suffix}))))')

        arr = self._get_array_iter_code(result)
        code = template.format(arr = arr,
                               op = op,
                               input2 = input2,
                               suffix = DTYPE_SUFFIXES[self._weld_type])
        result.weldobj.weld_code = code
        return result

    def _handle_reduce(self, ufunc, input_args, outputs, kwargs):
        '''
        TODO: describe.
        TODO: For multi-dimensional case, might need extra work here if the
        reduction is being performed only along a certain axis.
        We force evaluation at the end of the reduction - weird errors if we
        don't. This seems safer anyway.
        np supports reduce only for binary ops.
        '''
        if self._verbose:
            print('WARNING: not handling reduce because numpy seems to be faster')
        return None

        # input_args[0] must be self so it can be ignored.
        assert len(input_args) == 1
        if self._verbose: print('in handle reduce')
        if outputs: output = outputs[0]
        else: output = None
        axis = kwargs['axis']

        if ufunc.__name__ in wn.BINARY_OPS:
            return self._reduce_op(wn.BINARY_OPS[ufunc.__name__], axis=axis,
                    result=output)

    def evaluate(self):
        '''
        User facing function - if he wants to explicitly evaluate all the
        registered ops. Idempotent if no new ops have been registerd on the
        given array.
        '''
        ret = weldarray(self._eval(), verbose=self._verbose, new_weldobj=False)
        if self._weldarray_view:
            # Because it was an in place update, the shape, start/end information must be the same
            # as it was before evaluating (i.e, it can't change as it might happen if it was a
            # reduction etc.)
            ret._weldarray_view = self._weldarray_view
        return ret

    def _eval(self, restype=None):
        '''
        Internal call - used at various points to implicitly evaluate self. Users should instead
        call evaluate which would return a weldarray as expected. This returns an ndarray.

        @ret: ndarray after evaluating all the ops registered with the given weldarray.
        @restype: type of the result. Usually, it will be a WeldVec, but if called from reduction,
        it would be a scalar.
        Evalutes the expression based on weldobj.weld_code. If no new ops have been registered,
        then just returns the last ndarray.

        - View: If self is a view, then evaluates the parent array, and returns the aprropriate index from
        the result. Here, the behaviour is slightly different - it returns a weldarray
        '''
        # all registered ops will be cleared after this
        self._num_registered_ops = 0
        # This check has to happen before the caching - as the weldobj/code for
        # views is never updated.
        # TODO: Need to change this condition specifically for in place ops.
        # Case 1: we are evaluating an in place on in an array. So need to
        # evaluate the parent and return the appropriate index.
        if self._weldarray_view:
            idx = self._weldarray_view.idx
            strides = self._weldarray_view.strides
            if idx:
                # _eval parent and return appropriate idx.
                # TODO: more efficient to _eval the base array as the parent
                # would eventually have to do it - but this makes it more
                # convenient to deal with different indexing strategies.
                arr = self._weldarray_view.parent._eval()
                arr = arr[self._weldarray_view.idx]
                # FIXME: I don't think strides should be changed in all cases?
                if strides:
                    arr.strides = strides
                return arr
            else:
                # XXX hacky!! FIXME: need to have a better way and more general
                # to deal with transposes.
                if self._verbose: print('hacky eval transpose...')
                return self.view(np.ndarray)

        # Using cached results if no new ops have been added
        if self.name == self.weldobj.weld_code:
            # No new ops have been registered. Avoid creating unneccessary new
            # copies with weldobj.evaluate()
            ret = self.weldobj.context[self.name]
            return self.weldobj.context[self.name]

        if restype is None:
            # use default type for all weldarray operations
            restype = WeldVec(self._weld_type)
        arr = self.weldobj.evaluate(restype, verbose=self._verbose, passes=CUR_PASSES)

        if hasattr(arr, '__len__'):
            arr = arr.reshape(self._real_shape)
        else:
            # floats and other values being returned.
            return arr

        # Important to free memory here! Without this, some code like:
        #       >>> a = np.exp(a)
        #       >>> a = a.evaluate()
        # was leading to a huge memory leak as a's original array still had a
        # reference to the old array, so python's garbage collector did not
        # collect it for free-ing.
        del(self.weldobj.context[self.name])

        # Now that the evaluation is done - create new weldobject for self,
        # initalized from the returned arr. We want to be able to support users
        # not having to catch the return value to evaluate call - especially
        # since we do it many times internally as well.
        self._gen_weldobj(arr)
        return arr

    def _get_result(self):
        '''
        Creating a new result weldarray from self. If self is view into a
        weldarray, then evaluate the parent first as self would not be storing
        the ops that have been registered to it (only base_array would store
        those).
        '''
        if self._weldarray_view:
            idx = self._weldarray_view.idx
            if idx is not None:
                result = weldarray(self._weldarray_view.parent._eval()[idx],
                        verbose=self._verbose)
            else:
                result = weldarray(self)
        else:
            result = weldarray(self, verbose=self._verbose)
        return result

    def _get_array_iter_code(self, res):
        '''
        @self: weldarray; we want to loop over.
        @res: weldarray; going to be result after the op is applied on self.
        Need to pass it here so we can update res's context and dependencies.
        @ret: string; This represents the 'self' welarray's latest state - so
        we can use it in the code being added to res, e.g., for(ret, ......).
        Usually, we will just update it with the self objects weldobj.obj_id
        (e.g., obj100) and in weldobj's evaluate method, the weld code
        associated with the approrpiate obj_id is extracted into let
        statements:
            e.g., let obj100 = latest_code_of_obj100;
        But this breaks down if it is an in place operation - because let
        obj100 = obj100's code; will lead to an obvious an annoying recursion.
        So in this case, we will directly plug in obj100's code instead of
        obj_id.

        This is called at MANY places in the code. Typically, some operation
        (e.g., exp, or addition) is being performed on the weldarray self --
        and res is going to be returned after the operation has been
        registered. There are a few major scenarios:

            - self is a noncontiguous view: will have to use nditer.
            - self is a view, but is a contiguous chunk: This needs to be
              tested better / especially since in many cases we avoid calling
              this function entirely.
            - self is a conitiguous weldarray: This is the standard case which
              has been pretty well tested.
        '''
        # Checking if it is inplace is very important -- normally we would add the obj_id
        inplace = self.weldobj.obj_id == res.weldobj.obj_id
        if self._weldarray_view and not self.flags.contiguous:
            nditer_arr = 'nditer({arr},{start}L,{end}L,1L,{shape},{strides})'
            shape = np.array(self._weldarray_view.shape)
            strides = np.array(self._weldarray_view.strides)
            for i, s in enumerate(strides):
                strides[i] = s / self.itemsize

            shape = res.weldobj.update(shape)
            strides = res.weldobj.update(strides)
            # Note: the start/end/shapes/strides symbols for nditer are valid
            # if we start from the base array.
            arr = nditer_arr.format(arr = self._weldarray_view.base_array.weldobj.weld_code,
                                    start = self._weldarray_view.start,
                                    end = self._weldarray_view.end,
                                    shape = shape,
                                    strides = strides)
            res.weldobj.update(self._weldarray_view.base_array.weldobj)
            # TODO: Have a separate case for contiguous views. Check this
            # further -- maybe we can change self._get_result too?
            # could actually use nditer here as well but that is less
            # efficient...In this case, we can also change the else condition
            # to only be self.weldobj.weld_code.

        # contiguous flag implies that we don't need nditer here.
        elif self._weldarray_view and self.flags.contiguous:
            if inplace:
                assert False, 'this scenario should have been handled earlier'
                # arr = self.weldobj.weld_code
            else:
                # FIXME: NEED TO TEST THIS better. This should not work because
                # views DO NOT HAVE any operations stored in them.
                # arr = res.weldobj.obj_id
                arr = self.weldobj.obj_id
                res.weldobj.update(self.weldobj)
                res.weldobj.dependencies[self.weldobj.obj_id] = self.weldobj
        else:
            # Add id, and update dependencies for the returned array.
            assert isinstance(self, weldarray)
            if inplace:
                arr = self.weldobj.weld_code
            else:
                arr = self.weldobj.obj_id
                res.weldobj.update(self.weldobj)
                res.weldobj.dependencies[self.weldobj.obj_id] = self.weldobj
        return arr

    def _reduce_op(self, op, axis=None, result=None):
        '''
        TODO: support for multidimensional arrays.
        '''
        assert result is None, 'have not tested this yet'
        if axis is None:
            template = 'result(for({arr},merger[{type}, {op}], |b, i, e| merge(b, e)))'
            self.weldobj.weld_code = template.format(arr =
                    self.weldobj.weld_code, type = self._weld_type.__str__(),
                    op  = op)
            return self._eval(restype = self._weld_type)
        # reducing along an axis.
        # FIXME: Right now, only supporting the simplest of cases - 2d arrays,
        # with axis=0.  Loop over an outer appender, = dimensions of output -->
        # and then merge each row in the inner for loop.
        other_axis = (axis + 1) % 2
        template = ('result(for({dim_arr}, appender, |b,i,e| merge(b, result(for(iter({arr},'
                    '{start},{end}L,{stride}L), merger[{type}, {op}], |b2, i2, e2| merge(b2,'
                    'e2))))))')
        rows = self.shape[axis]
        columns = self.shape[other_axis]
        outer_stride = self.strides[axis] / self.itemsize
        inner_stride = self.strides[other_axis] / self.itemsize

        start = 'i*{}L'.format(inner_stride)
        stride = str(outer_stride)
        # FIXME: Technically, I think this should work as well but is it a bug
        # in weld? It seems like iter(e0, 0, 3, 4) would fail because end
        # should be exactly 4 or above.  end =
        # str(outer_stride*self.shape[0]+1)
        end = 'i*{}L + {}L*{}'.format(inner_stride, outer_stride, rows)
        dim_arr = np.array(range(self.shape[other_axis]))
        # declare an output array
        # TODO: is this best way?
        result = self._get_result()
        result._real_shape = (self.shape[other_axis],)

        arr = self._get_array_iter_code(result)
        dim_arr_name = result.weldobj.update(dim_arr, SUPPORTED_DTYPES[str(dim_arr.dtype)])
        code = template.format(dim_arr = dim_arr_name,
                               arr = arr,
                               start = start,
                               end = end,
                               stride = stride,
                               type = self._weld_type.__str__(),
                               op = op)
        result.weldobj.weld_code = code
        return result

    def _power_op(self, power, result=None):
        '''
        TODO: support for multidimensional arrays.
        '''
        if result is None:
            result = self._get_result()
        else:
            assert False, 'TODO: support this'
        template = 'result(for({arr}, appender, |b,i,e| merge(b,powi(e,{pow}))))'
        arr = self._get_array_iter_code(result)
        result.weldobj.weld_code = template.format(arr = arr,
                                                  pow = power)
        return result

    def _unary_op(self, unop, result=None):
        '''
        @unop: str, weld IR for the given function.
        @result: output array - if it is an inplace op then this must be specified already.
        @ret: the result array. If result is not None, then the same array will be returned after
        the new operation is registered with it.
        '''
        def _update_array_unary_op(res, unop):
            '''
            @res: weldarray to be updated.
            @unop: str, operator applied to res.
            '''
            template = 'result(for({arr}, appender,|b,i,e| merge(b,{unop}(e))))'
            arr = self._get_array_iter_code(res)
            code = template.format(arr = arr, unop = unop)
            res.weldobj.weld_code = code

        if result is None:
            result = self._get_result()
        else:
            if not isinstance(result, weldarray):
                return None
            # TODO: make general _get_result() function for in place ops.
            # in place op + view, just update base array and return.
            if result._weldarray_view:
                v = result._weldarray_view
                update_str = '{unop}(e)'.format(unop=unop)
                v.base_array._update_range(v.start, v.end, update_str)
                return result

        # back to updating result array
        # Since it is a unary op - result is derived from self, so this is not
        # required. But if we change the way result is derived - i.e., use
        # something like np.empty(shape) - then we will need this.
        result.weldobj.update(self.weldobj)
        _update_array_unary_op(result, unop)
        return result

    def _update_range(self, start, end, update_str, strides=1):
        '''
        @start, end: define which values of the view needs to be updated - for
        a child, it would be all values, and for parent it wouldn't be.
        @update_str: code to be executed in the if block to update the
        variable, 'e', in the given range.
        '''
        views_template = "result(for({arr1}, appender,|b,i,e| if \
        (i >= {start}L & i < {end}L,merge(b,{update_str}),merge(b,e))))"

        # all values of child will be updated. so start = 0, end = len(c)
        self.weldobj.weld_code = views_template.format(arr1 = self.weldobj.weld_code,
                                                      start = start,
                                                      end   = end,
                                                      update_str = update_str)

    def _binary_op(self, input1, input2, binop, result=None):
        '''
        @input1: weldarray, arg 1.
        @input2: weldarray, arg 2.
        @result: output array. If it has been specified by the caller, then
        don't allocate new array. This MUST be input1, or input2.

        TODO: Explain Scenario 1 (non-inplace op) and 4 cases + Scenario 2
        (inplace op) and its cases and how we deal with them.
        '''
        def _update_input(inp):
            '''
            TODO: can we avoid these?
            1. evaluate the parent if it is a view.
            2. convert to weldarray if it is a numpy ndarray.
            '''
            if isinstance(inp, np.ndarray) and not isinstance(inp, weldarray):
                # TODO: could optimize this by skipping conversion to weldarray.
                # turn it into weldarray - for convenience, as this reduces to the
                # case of other being a weldarray.
                inp = weldarray(inp)
            return inp

        def _scalar_binary_op(input1, input2, binop, result):
            '''
            Helper function for _binary_op.
            @other, scalar values (i32, i64, f32, f64).
            @result: weldarray to store results in.
            '''
            # which of these is the scalar?
            if not hasattr(input2, "__len__"):
                template = ('result(for({arr}, appender,|b,i,e| merge(b,e {binop}'
                '{scalar}{suffix})))')
                arr = input1._get_array_iter_code(result)
                scalar = input2
                real_shape = input1._real_shape
            else:
                template = ('result(for({arr}, appender,|b,i,e| merge(b,{scalar}{suffix} {binop}'
                'e)))')
                arr = input2._get_array_iter_code(result)
                scalar = input1
                real_shape = input2._real_shape

            weld_type = result._weld_type.__str__()
            result.weldobj.weld_code = template.format(arr = arr,
                                                      binop = binop,
                                                      scalar = str(scalar),
                                                      suffix = DTYPE_SUFFIXES[weld_type])
            result._real_shape = real_shape
            return result

        def _update_views_binary(result, other, binop):
            '''
            @result: weldarray that is being updated
            @other: weldarray or scalar. (result binop other)
            @binop: str, operation to perform.

            FIXME: the common indexing pattern for parent/child in _update_view might be too expensive
            (uses if statements (unneccessary checks when updating child) and wouldn't be ideal to
            update a large parent).
            '''
            # we should not update result as that would evaluate the parent array - since we are
            # only adding more inplace ops, that should not be needed.
            other = _update_input(other)
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

        def broadcast_arrays(arr1, arr2):
            '''
            If arr1.shape != arr2.shape, then this tries to use numpy.broadcast to still do the
            computation.
            '''
            def finalize_array(arr, orig_arr):
                '''
                Creates a weldview only if orig_arr is actually different from arr.
                We seeem to need this for some slightly mysterious reason.
                TODO: might need to check edge cases for nditer.
                Solves the issue with test_broadcasting_bug.
                '''
                if arr.shape != orig_arr.shape:
                    return orig_arr._gen_weldview(arr)
                else:
                    return orig_arr
            arr1_b, arr2_b = np.broadcast_arrays(arr1, arr2)
            arr1_b = finalize_array(arr1_b, arr1)
            arr2_b = finalize_array(arr2_b, arr2)
            return arr1_b, arr2_b

        # Scenario 1: Inplace Op
        if result is not None and result._weldarray_view:
            # which one of inputs is result?
            if isinstance(input1, weldarray) and input1.name == result.name:
                _update_views_binary(result, input2, binop)
            else:
                assert input2.name == result.name, 'has to be the case'
                _update_views_binary(result, input1, binop)
            return result

        # Scenario 2b: Non-Inplace ops.
        if result is None:
            result = self._get_result()
        # if it is not None, then result must be a weldarray already - and
        # since we are updating it in place, then we should not call
        # weldarray(result) - or this would change the weldobj.obj_id for the
        # array.
        assert isinstance(result, weldarray)

        # Scenario 2: Non Inplace Op + Inplace Op on a non-view.
        input1 = _update_input(input1)
        input2 = _update_input(input2)

        # Broadcasting arrays here because it (probably?) won't work with the
        # way we're doing inplace ops.
        if isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):
            # we need to compare real shapes because the operations are lazily
            # evaluated so shapes could be changing...
            if (input1._real_shape != input2._real_shape):
                if self._verbose: print('broadcasting!')
                # need to broadcast the arrays!
                input1, input2 = broadcast_arrays(input1, input2)

        # scalars (i32, i64, f32, f64...)
        if not hasattr(input1, "__len__") or not hasattr(input2, "__len__"):
            _scalar_binary_op(input1, input2, binop, result)
        else:
            # update result's weldobj with the arrays it needs. We can be sure
            # that both the inputs are weldarrays by this point.
            arr1 = input1._get_array_iter_code(result)
            arr2 = input2._get_array_iter_code(result)
            template2 = ('result(for(zip({arr1},{arr2}), appender,|b,i,e| merge(b,e.$0'
            ' {binop} e.$1)))')
            result.weldobj.weld_code = template2.format(arr1 = arr1,
                                                        arr2  = arr2,
                                                        binop = binop)
            # Important to have the correct shape. Since we have already done
            # broadcast, this must be right.
            assert input1._real_shape == input2._real_shape, 'test shapes of inputs'
            result._real_shape = input1._real_shape
        return result
