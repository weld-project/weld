from weld.weldobject import *
from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
from weldnumpy import *
import time

class weldarray(np.ndarray):
    '''
    A new weldarray can be created in three ways:
        - Explicit constructor. This is dealt with in __new__
        - Generating view of weldarray (same as ndarray)
        - fancy indexing (same as ndarray)
    The second and third case do not pass through __new__. Instead, numpy
    guarantees that after initializing an array, in all three methods, numpy
    will call __array_finalize__. But we are able to deal with the 2nd and 3rd
    case in __getitem__ so there does not seem to be any need to use
    __array_finalize (besides __array_finalize__ also adds a function call to
    the creation of a new array, which adds to the overhead as compared to
    numpy for initializing arrays)
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
        obj.view_siblings = []

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
        TODO: Multidimensional support. Will make it a lot more complicated.
        a[...] type of indexing.
        Deal with the following three scenarios:
            - idx is a scalar
            - idx is a slice (e.g., 3:5)
                - In this scenario, if the memory is being shared with the
                  parent then we need to update child/parent views lists.
            - idx is an array (fancy indicing)

        TODO: Describe parents, children, siblings.
        Steps:
            1. Evaluate all registered ops. This is definitely neccessary for
            the first and the third scenario. In scenario 2, it may be possible
            to avoid it - but that leads to a lot of edge case:
                - e.g., parent's registered ops would need to be applied to
                  child. Simple solution of including parents weldcode in child
                  does not work as the parent might have index specific weld
                  code etc.
            2. Call np.ndarray on the evaluated array.
            3. Convert back to weldarray if needed.
        '''
        # evaluate all lazily stored ops, if any.
        # Need to cast it as ndarray view, as otherwise arr.base will be a
        # weld_encoders object - so fails in is_view_child.
        arr = self._eval().view(np.ndarray)
        # arr is an ndarray, so we call ndarray's implementation on it.
        ret = arr.__getitem__(idx)

        # handle views in this block.
        # TODO: decompose / improve this chunk.
        if isinstance(idx, slice):
            ret = weldarray(ret)
            # check if ret really is a view of arr.
            if is_view_child(ret.view(np.ndarray), arr):
                # we need to calculate start / end right now, because otherwise
                # registering / evaluating ops on the parent in the future will
                # change its address (see test_views_update_mix in tests_1d for
                # an example)
                ret_start = (addr(ret.view(np.ndarray)) - addr(arr)) / ret.itemsize
                ret_end = ret_start + len(ret)

                # TODO: explain ordering.
                # Update sibling relationship between ret and self's children.
                for c in self.view_children:
                    # check if the children of self have overlap in parent indices
                    # x1 < y2 and y1 < x2, with all indices relative to parent array.
                    if (ret_start < (c.other_start+c.upd_end) and c.other_start < ret_end):
                        print("found sibling!")
                        # TODO: decompose the finding of indices.
                        # c is a sibling of ret. find index range in which c should be updated.
                        c_start = max(0, ret_start - c.other_start)
                        c_end = min(c.other_start+c.upd_end, ret_end) - c.other_start
                        assert c_start < c_end, 'start < end'
                        # where in ret does c start?
                        ret_c_start = max(0, c.other_start-ret_start)
                        assert ret_c_start >= 0, 'ret_c_start > 0'
                        # c_start, c_end are indices of c, where ret starts
                        ret.view_siblings.append(weldarray_view(c, c_start, c_end, ret_start))

                        r_start = max(0, c.other_start-ret_start)
                        r_end = min(ret_end, c.other_start+c.upd_end) - ret_start
                        c_ret_start = max(0, ret_start - c.other_start)
                        # add c to ret's sibling list.
                        c.arr.view_siblings.append(weldarray_view(ret, r_start, r_end, c_ret_start))

                # Update parent-child relationship between self / and new view
                # add self's children. all of ret should be updated if self (parent) is updated.
                self.view_children.append(weldarray_view(ret, 0, len(ret), ret_start))
                # add child's parent. upd_start = index in parent (self) where the child starts.
                # other_start will always be 0 because all of child in parent.
                ret.view_parents.append(weldarray_view(self, ret_start, ret_end, 0))

                # parenting is transitive, i.e., if x is parent of y, and z is
                # parent of x, then z is also parent of y.
                # Note: we need to do this after updating siblings first, otherwise TODO: Explain
                # problem.
                for p in self.view_parents:
                    gp_start = p.upd_start + ret_start
                    gp_end = p.upd_start + ret_end
                    print("gp_start = {}, gp_end = {}".format(gp_start, gp_end))
                    p.arr.view_children.append(weldarray_view(ret, 0, len(ret), gp_start))
                    ret.view_parents.append(weldarray_view(p.arr, gp_start, gp_end, 0))


            return ret

        elif isinstance(idx, np.ndarray):
            # this will not have shared memory with parent array.
            return weldarray(ret)
        else:
            # return the scalar.
            return ret

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
            # FIXME: This fails if output is ndarray!
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

        # check for supported ops.
        if ufunc.__name__ in UNARY_OPS and supported:
            assert(len(input_args) == 1)
            return self._unary_op(UNARY_OPS[ufunc.__name__], result=output)

        if ufunc.__name__ in BINARY_OPS and supported:
            # self can be first or second arg.
            if isinstance(input_args[0], weldarray):
                # first arg is weldarray, must be self
                assert input_args[0].name == self.name
                other_arg = input_args[1]
            else:
                other_arg = input_args[0]
                assert input_args[1].name == self.name

            return self._binary_op(other_arg, BINARY_OPS[ufunc.__name__],
                    result=output)

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
        if str(result.dtype) in SUPPORTED_DTYPES:
            return weldarray(result)
        else:
            return result

    def evaluate(self):
        '''
        User facing function - if he wants to explicitly evaluate all the
        registered ops.
        '''
        return weldarray(self._eval())

    def _eval(self):
        '''
        @ret: ndarray after evaluating all the ops registered with the given
        weldarray.

        Users should instead call evaluate which would return a weldarray as
        expected. This returns an ndarray.

        Evalutes the expression based on weldobj.weld_code. If no new ops have
        been registered, then just returns the last ndarray.

        TODO: Cache views for parents/children. Caching computations for
        children is easy, just update the whole array. Becomes messier for
        parents - might need to store additional info in the weldview objects
        for parents.
        '''
        if self.name == self.weldobj.weld_code:
            # no new ops have been registered. Let's avoid creating
            # unneccessary new copies with weldobj.evaluate()
            return self.weldobj.context[self.name]

        arr = self.weldobj.evaluate(WeldVec(self._weld_type),
                verbose=self.verbose)
        # Now that the evaluation is done - create new weldobject for self,
        # initalized from the given arr.
        self._gen_weldobj(arr)
        return arr

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
            code = template.format(arr = res.weldobj.weld_code,
                    type=res._weld_type.__str__(),
                    unop=unop)
            res.weldobj.weld_code = code

        if result is None:
            result = weldarray(self)

        # Update results children/parents because results represent the output
        # array, ie. the array that will be changed in place.
        for c in result.view_children:
            # updating child is easy - just apply unary op to every
            # element.
            _update_array_unary_op(c.arr, unop)

        for p in result.view_parents:
            # print("updating unary op parent!")
            # print("p_us: {}, p_ue: {}, p_os: {}".format(p.upd_start, p.upd_end, p.other_start))
            par_template = "result(for({arr}, appender,|b,i,e| if \
            (i >= {start}L & i < {end}L,merge(b,{unop}(e)),merge(b,e))))"
            p.arr.weldobj.weld_code = par_template.format(arr=p.arr.weldobj.weld_code,
                                       start = str(p.upd_start),
                                       end = str(p.upd_end),
                                       unop=unop)

        # TODO: refactor this stuff with update parents
        for s in result.view_siblings:
            # apply unop to s.upd_start, end
            print("updating siblings in unary op")
            sib_template = "result(for({arr}, appender,|b,i,e| if \
            (i >= {start}L & i < {end}L,merge(b,{unop}(e)),merge(b,e))))"
            s.arr.weldobj.weld_code = sib_template.format(arr=s.arr.weldobj.weld_code,
                                       start = str(s.upd_start),
                                       end = str(s.upd_end),
                                       unop=unop)
        # back to updating result array
        _update_array_unary_op(result, unop)
        return result

    def _scalar_binary_op(self, other, binop, result):
        '''
        Helper function for _binary_op.
        @other is a scalar (i32, i64, f32, f64).
        @result: None or a weldarray to store results in.
        '''
        if result is None:
            result = weldarray(self)
        else:
            self._update_views(result, other, binop)

        template = 'map({arr}, |z: {type}| z {binop} {other}{suffix})'
        result.weldobj.weld_code = template.format(arr = result.weldobj.weld_code,
                                                  type = result._weld_type.__str__(),
                                                  binop = binop,
                                                  other = str(other),
                                                  suffix = DTYPE_SUFFIXES[str(type(other))])
        return result

    def _update_views(self, result, other, binop):
        '''
        @result: weldarray that is being updated
        @other: weldarray. (result binop other)
        @binop: str, operation to perform.

        Will go over any parent/child of result and update them accordingly.

        FIXME: the common indexing pattern for parent/child in _update_view
        might be too expensive (uses if statements (unneccessary checks when
        updating child) and wouldn't be ideal to update a really long parent).
        '''
        def _update_view(view, lookup_ind, start, end):
            '''
            @view: weldarray that needs to be updated. could be a parent or a
            child view.
            @lookup_ind: int, used to update the correct values in view. Will
            be different if view is a child or parent array.
            @start, end: define which values of the view needs to be updated -
            for a child, it would be all values, and for parent it wouldn't be.
            '''
            views_template = "result(for({arr1}, appender,|b,i,e| if \
            (i >= {start}L & i < {end}L,merge(b,{e2}{binop}e),merge(b,e))))"

            if isinstance(other, weldarray):
                view.weldobj.update(other.weldobj)
                e2 = 'lookup({arr2},{i}L)'.format(arr2 = other.weldobj.weld_code,
                                                         i = lookup_ind)
            else:
                # other is just a scalar.
                e2 = str(other) + DTYPE_SUFFIXES[str(type(other))]

            # all values of child will be updated. so start = 0, end = len(c)
            view.weldobj.weld_code = views_template.format(arr1 = view.weldobj.weld_code,
                                                          start = start,
                                                          end = end,
                                                          e2=e2,
                                                          binop=binop)
        for c in result.view_children:
            _update_view(c.arr, 'i+{st}'.format(st=c.other_start), c.upd_start, c.upd_end)

        for p in result.view_parents:
            _update_view(p.arr, 'i-{st}'.format(st=p.upd_start), p.upd_start, p.upd_end)

        for s in result.view_siblings:
            print("going to update sibling in binaryop")
            print("s.os = {}, s.us = {}, s.ue = {}".format(s.other_start, s.upd_start, s.upd_end))
            if s.other_start == 0:
                _update_view(s.arr, 'i-{st}'.format(st=s.upd_start), s.upd_start, s.upd_end)
            else:
                print("updating non-os = 0 array")
                _update_view(s.arr, 'i+{st}'.format(st=s.other_start), s.upd_start, s.upd_end)

    def _binary_op(self, other, binop, result=None):
        '''
        @other: ndarray, weldarray or scalar.
        @binop: str, one of the weld supported binop.
        @result: output array. If it has been specified by the caller, then
        don't allocate new array.
        '''
        if not hasattr(other, "__len__"):
            return self._scalar_binary_op(other, binop, result)

        is_weld_array = False
        if isinstance(other, weldarray):
            is_weld_array = True
        elif isinstance(other, np.ndarray):
            # turn it into weldarray - for convenience, as this reduces to the
            # case of other being a weldarray.
            # FIXME: could optimize this by skipping conversion.
            other = weldarray(other)
        else:
            raise ValueError('invalid array type for other in binaryop')

        if result is None:
            result = weldarray(self)
            # in this case, we do not need to change values in parent/child
            # arrays because new array is created so there won't be any views.
        else:
            self._update_views(result, other, binop)

        result.weldobj.update(other.weldobj)
        # @arr{i}: latest array based on the ops registered on operand{i}. {{
        # is used for escaping { for format.
        template = 'map(zip({arr1},{arr2}), |z: {{{type1},{type2}}}| z.$0 {binop} z.$1)'
        result.weldobj.weld_code = template.format(arr1 = result.weldobj.weld_code,
                                                  arr2 = other.weldobj.weld_code,
                                                  type1 = result._weld_type.__str__(),
                                                  type2 = other._weld_type.__str__(),
                                                  binop = binop)

        if not is_weld_array:
            # FIXME: Not completely sure this is neccessary. Recheck the
            # np.allclose example.
            # Evaluate all the lazily stored ops, as the caller probably
            # expects to get back a ndarray - which would have correct values.
            result = weldarray(result._eval())

        return result

