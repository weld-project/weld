"""
Aggregation functions implemented in Weld.

Supports co-generation of aggregation functions to enable better optimization.

"""

import copy
import weld.lazy

from weld.types import I64, F32, F64

def supported(name):
    """
    Returns whether codegen for this aggregation is supported.

    Tests
    -----

    >>> supported('min')
    True
    >>> supported('humdinger')
    False

    """
    return name in _agg_supported

def result_elem_type(input_ty, aggs):
    """
    Return the resulting element type of a list of aggregations.

    This will either be 'I64' or 'F64'.
    """
    aggs = sorted(aggs, key=lambda x: _agg_priorities[x])
    for agg in aggs:
        input_ty = _agg_output_types[agg](input_ty)
    return input_ty

@weld.lazy.weldfunc
def agg(array, weld_elem_type, aggs):
    """
    Compute aggregations with Weld.

    Aggregation functions with Weld always return either a 'i64' or 'f64',
    depending on the operations passed. If any operation passed requires use of
    a 'f64', all outputs will be 'f64'.

    The return type of an aggregation will be a scalar if exactly one function
    was passed, or a vector if more than one was passed. The output element
    type can be retrieved with 'result_elem_type()'.

    """
    aggs_to_run = set(aggs)
    for agg in aggs:
        # Add all the dependencies.
        aggs_to_run = aggs_to_run.union(dependencies_recursive(agg))

    # Compute aggregations in order of priorty, and remove duplicates.
    sorted_aggs = sorted(list(aggs_to_run), key=lambda x: _agg_priorities[x])
    # Names referring to already-computed aggs, keyed by agg name.
    deps = dict()
    # Computations, keyed by agg name
    computations = dict()
    previous_output_type = weld_elem_type
    # Final result program.
    result = ""

    for agg in sorted_aggs:
        # Use the element type produced by the aggregation.
        # Inputs are cast to this type.
        elem_type = _agg_output_types[agg](previous_output_type)
        code = _agg_funcs[agg](array, str(elem_type), deps)
        name = "{}_result".format(agg)
        deps[agg] = name
        computations[agg] = code
        previous_output_type = elem_type
        # Append the just-computed agg to the final code.
        result += "let {name} = {code};\n".format(name=name, code=code)

    if len(aggs) == 1:
        # Just return the scalar if there is only one aggregation requested.
        # Note that we may have computed more than one if there were dependencies.
        result += deps[aggs[0]]
        return result
    else:
        # If there are multiple aggregations, construct a vector to hold them. The vector
        # needs to be ordered in the same way that the aggregations were requested in, and they
        # need to be cast to the same type.
        output_type = result_elem_type(weld_elem_type, aggs)
        ordered_results = []
        # Iterate in the original order. This could have duplicates, which is okay.
        for agg in aggs:
            ordered_results.append("{ty}({name})".format(ty=output_type, name=deps[agg]))
        result += "[" + ", ".join(ordered_results) + "]"
        return result

def dependencies_recursive(agg):
    """
    Recursively get all dependencies of 'agg'.

    Tests
    -----

    >>> sorted(list(dependencies_recursive('var')))
    ['count', 'mean', 'sum']

    """
    deps = set(_agg_dependencies[agg])
    for dep_agg in _agg_dependencies[agg]:
        deps = deps.union(dependencies_recursive(dep_agg))
    return deps


def agg_template(array, agg_func, elem_type):
    """
    Return Weld code for computing an aggregation with a merger.

    """
    return """result(for({array}, merger[{ty},{agg_func}],
    |b, i, e|
        merge(b, {ty}(e))
    ))""".format(array=array,
            agg_func=agg_func,
            ty=elem_type)


def count(weldarr, _elem_type, _deps):
    """
    Computes the count.

    Tests
    -----
    >>> count('[1,2,3]', None, None)
    'len([1,2,3])'
    >>> count('weldlazy1', None, None)
    'len(weldlazy1)'

    """
    return "len({})".format(weldarr)


def weld_min(array, elem_type, _deps):
    """
    Computes the min.

    Tests
    -----
    >>> weld_min('array', 'i64', None)
    'result(for(array, merger[i64,min],\\n    |b, i, e|\\n        merge(b, i64(e))\\n    ))'

    """
    return agg_template(array, 'min', elem_type)


def weld_max(array, elem_type, _deps):
    """
    Computes the max.

    Tests
    -----
    >>> weld_max('array', 'i64', None)
    'result(for(array, merger[i64,max],\\n    |b, i, e|\\n        merge(b, i64(e))\\n    ))'

    """
    return agg_template(array, 'max', elem_type)


def weld_sum(array, elem_type, _deps):
    """
    Computes the max.

    Tests
    -----
    >>> weld_sum('array', 'i64', None)
    'result(for(array, merger[i64,+],\\n    |b, i, e|\\n        merge(b, i64(e))\\n    ))'

    """
    return agg_template(array, '+', elem_type)


def prod(array, elem_type, _deps):
    """
    Computes the product.

    Tests
    -----
    >>> prod('array', 'i64', None)
    'result(for(array, merger[i64,*],\\n    |b, i, e|\\n        merge(b, i64(e))\\n    ))'

    """
    return agg_template(array, '*', elem_type)


def mean(_array, elem_type, deps):
    """
    Computes the mean.

    deps must be a dictionary with 'sum' and 'count'.

    Tests
    -----
    >>> mean(None, 'f64', {'sum': 'weldlazy1', 'count': 'weldlazy2'})
    'f64(weldlazy1) / f64(weldlazy2)'

    """
    return """{ty}({sum}) / {ty}({count})""".format(
            ty=elem_type,
            sum=deps['sum'],
            count=deps['count'])


def var(array, _elem_type, deps):
    """
    Computes the variance.

    deps must be a dictionary with 'count' and 'mean'.

    Tests
    -----
    >>> var('array', 'f64', {'count': 'weldlazy1', 'mean': 'weldlazy2'})
    'result(for(array,\\n        merger[f64, +],\\n        |b, i, e|\\n             merge(b, pow(f64(e) - f64(weldlazy2), 2.0))\\n    )) / f64(weldlazy1 - 1L)'

    """
    return """result(for({array},
        merger[f64, +],
        |b, i, e|
             merge(b, pow(f64(e) - f64({mean}), 2.0))
    )) / f64({count} - 1L)""".format(
            array=array,
            count=deps['count'],
            mean=deps['mean'])

def std(array, _elem_type, deps):
    """
    Computes the standard deviation.

    deps must be a dictionary with 'var'.

    Tests
    -----
    >>> std('array', None, {'var': 'weldlazy1'})
    'sqrt(weldlazy1)'

    """
    return "sqrt({var})".format(var=deps['var'])




_agg_supported = {
    'min',
    'max',
    'count',
    'sum',
    'prod',
    'mean',
    'var',
    'std',
}

_agg_funcs = {
    'min': weld_min,
    'max': weld_max,
    'count': count,
    'sum': weld_sum,
    'prod': prod,
    'mean': mean,
    'var': var,
    'std': std,
}

_agg_dependencies = {
    'min': set(),
    'max': set(),
    'count': set(),
    'sum': set(),
    'prod': set(),
    'mean': {'sum', 'count'},
    'var': {'count', 'mean'},
    'std': {'var'}
}

# to order the aggregations; lower means it comes first
_agg_priorities = {
    'min': 1,
    'max': 1,
    'count': 1,
    'sum': 1,
    'prod': 1,
    'mean': 2,
    'var': 3,
    'std': 4
}

# Type specifiers
_always_i64 = lambda _: I64()
_i64_or_f64 = lambda x: F64() if isinstance(x, (F32, F64)) else I64()
_always_f64 = lambda _: F64()

_agg_output_types = {
    'count': _always_i64,
    'min': _i64_or_f64,
    'max': _i64_or_f64,
    'sum': _i64_or_f64,
    'prod': _i64_or_f64,
    'mean': _always_f64,
    'var': _always_f64,
    'std': _always_f64,
}


# Assertions to catch mistakes in supported functions.
assert _agg_supported == set(_agg_priorities.keys())
assert _agg_supported == set(_agg_dependencies.keys())
assert _agg_supported == set(_agg_output_types.keys())
assert _agg_supported == set(_agg_funcs.keys())
