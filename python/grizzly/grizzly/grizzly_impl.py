"""Contains implementations for each ported operation in Pandas.

Attributes:
    decoder_ (NumPyEncoder): Description
    encoder_ (NumPyDecoder): Description
"""
from encoders import *
from weld.weldobject import *


encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()

def get_field(expr, field):
    """ Fetch a field from a struct expr

    """
    weld_obj = WeldObject(encoder_, decoder_)

    struct_var = weld_obj.update(expr)
    if isinstance(expr, WeldObject):
        struct_var = expr.obj_id
        weld_obj.dependencies[struct_var] = expr

    weld_template = """
      %(struct)s.$%(field)s
    """

    weld_obj.weld_code = weld_template % {"struct":struct_var,
                                          "field":field}
    return weld_obj

def unique(array, ty):
    """
    Returns a new array-of-arrays with all duplicate arrays removed.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
       map(
         tovec(
           result(
             for(
               map(
                 %(array)s,
                 |p: %(ty)s| {p,0}
               ),
               dictmerger[%(ty)s,i32,+],
               |b, i, e| merge(b,e)
             )
           )
         ),
         |p: {%(ty)s, i32}| p.$0
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "ty": ty}
    return weld_obj


def aggr(array, op, initial_value, ty):
    """
    Returns sum of elements in the array.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        op (str): Op string used to aggregate the array (+ / *)
        initial_value (int): Initial value for aggregation
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
      result(
        for(
          %(array)s,
          merger[%(ty)s,%(op)s],
          |b, i, e| merge(b, e)
        )
      )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var, "ty": ty, "op": op}
    return weld_obj


def mask(array, predicates, new_value, ty):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        new_value (WeldObject / Numpy.ndarray / str): mask value
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates

    if str(ty).startswith("vec"):
        new_value_var = weld_obj.update(new_value)
        if isinstance(new_value, WeldObject):
            new_value_var = new_value.obj_id
            weld_obj.dependencies[new_value_var] = new_value
    else:
        new_value_var = "%s(%s)" % (ty, str(new_value))

    weld_template = """
       map(
         zip(%(array)s, %(predicates)s),
         |p: {%(ty)s, bool}| if (p.$1, %(new_value)s, p.$0)
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var,
        "predicates": predicates_var,
        "new_value": new_value_var,
        "ty": ty}

    return weld_obj


def filter(array, predicates, ty=None):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates

    weld_template = """
       result(
         for(
           zip(%(array)s, %(predicates)s),
           appender,
           |b, i, e| if (e.$1, merge(b, e.$0), b)
         )
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var,
        "predicates": predicates_var}

    return weld_obj

def pivot_filter(pivot_array, predicates, ty=None):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_array_var = weld_obj.update(pivot_array)
    if isinstance(pivot_array, WeldObject):
        pivot_array_var = pivot_array.obj_id
        weld_obj.dependencies[pivot_array_var] = pivot_array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates

    weld_template = """
    let index_filtered =
      result(
        for(
          zip(%(array)s.$0, %(predicates)s),
          appender,
          |b, i, e| if (e.$1, merge(b, e.$0), b)
        )
      );
    let pivot_filtered =
      map(
        %(array)s.$1,
        |x|
          result(
            for(
              zip(x, %(predicates)s),
              appender,
              |b, i, e| if (e.$1, merge(b, e.$0), b)
            )
          )
      );
    {index_filtered, pivot_filtered, %(array)s.$2}
    """
    weld_obj.weld_code = weld_template % {
        "array": pivot_array_var,
        "predicates": predicates_var}

    return weld_obj

def isin(array, predicates, ty):
    """ Checks if elements in array is also in predicates
    
    # TODO: better parameter naming

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates
    weld_template = """
    let check_dict =
      result(
        for(
          map(
            %(predicate)s,
            |p: %(ty)s| {p,0}
          ),
        dictmerger[%(ty)s,i32,+],
        |b, i, e| merge(b,e)
        )
      );
      map(
        %(array)s,
        |x: %(ty)s| keyexists(check_dict, x)
      )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var,
        "predicate": predicates_var,
        "ty": ty}

    return weld_obj


def element_wise_op(array, other, op, ty):
    """
    Operation of series and other, element-wise (binary operator add)

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second Input array
        op (str): Op string used to compute element-wise operation (+ / *)
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    other_var = weld_obj.update(other)
    if isinstance(other, WeldObject):
        other_var = other.obj_id
        weld_obj.dependencies[other_var] = other

    weld_template = """
       map(
         zip(%(array)s, %(other)s),
         |a| a.$0 %(op)s a.$1
       )
    """

    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "other": other_var,
                                          "ty": ty, "op": op}
    return weld_obj

def unzip_columns(expr, column_types):
    """
    Zip together multiple columns.

    Args:
        columns (WeldObject / Numpy.ndarray): lust of columns

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    column_appenders = []
    struct_fields = []
    result_fields = []
    for i, column_type in enumerate(column_types):
        column_appenders.append("appender[%s]" % column_type)
        struct_fields.append("merge(b.$%s, e.$%s)" % (i, i))
        result_fields.append("result(unzip_builder.$%s)" % i)
    appender_string = "{%s}" % ", ".join(column_appenders)
    struct_string = "{%s}" % ", ".join(struct_fields)
    result_string = "{%s}" % ", ".join(result_fields)
    expr_var = weld_obj.update(expr)
    if isinstance(expr, WeldObject):
        expr_var = expr.obj_id
        weld_obj.dependencies[expr_var] = expr

    weld_template = """
    let unzip_builder = for(
      %(expr)s,
      %(appenders)s,
      |b,i,e| %(struct_builder)s
    );
    %(result)s
    """

    weld_obj.weld_code = weld_template % {"expr": expr_var,
                                          "appenders": appender_string,
                                          "struct_builder": struct_string,
                                          "result": result_string}
    return weld_obj

def sort(expr, field = None, keytype=None, ascending=True):
    """
    Sorts the vector.
    If the field parameter is provided then the sort
    operators on a vector of structs where the sort key
    is the field of the struct.

    Args:
      expr (WeldObject)
      field (Int)
    """
    weld_obj = WeldObject(encoder_, decoder_)

    expr_var = weld_obj.update(expr)
    if isinstance(expr, WeldObject):
        expr_var = expr.obj_id
        weld_obj.dependencies[expr_var] = expr

    if field is not None:
        key_str = "x.$%s" % field
    else:
        key_str = "x"

    if not ascending:
        # The type is not necessarily f64.
        key_str = key_str + "* %s(-1)" % keytype

    weld_template = """
    sort(%(expr)s, |x| %(key)s)
    """
    weld_obj.weld_code = weld_template % {"expr":expr_var, "key":key_str}
    return weld_obj

def slice_vec(expr, start, stop):
    """
    Slices the vector.

    Args:
      expr (WeldObject)
      start (Long)
      stop (Long)
    """
    weld_obj = WeldObject(encoder_, decoder_)

    expr_var = weld_obj.update(expr)
    if isinstance(expr, WeldObject):
        expr_var = expr.obj_id
        weld_obj.dependencies[expr_var] = expr

    weld_template = """
    slice(%(expr)s, %(start)sL, %(stop)sL)
    """
    weld_obj.weld_code = weld_template % {"expr":expr_var,
                                          "start":start,
                                          "stop":stop}
    return weld_obj

def zip_columns(columns):
    """
    Zip together multiple columns.

    Args:
        columns (WeldObject / Numpy.ndarray): lust of columns

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    column_vars = []
    for column in columns:
        col_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            col_var = column.obj_id
            weld_obj.dependencies[col_var] = column
        column_vars.append(col_var)

    arrays = ", ".join(column_vars)

    weld_template = """
       result(
         for(
           zip(%(array)s),
           appender,
           |b, i, e| merge(b, e)
         )
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": arrays,
    }

    return weld_obj

def compare(array, other, op, ty_str):
    """
    Performs passed-in comparison op between every element in the passed-in
    array and other, and returns an array of booleans.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second input array
        op (str): Op string used for element-wise comparison (== >= <= !=)
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    # Strings need to be encoded into vec[char] array.
    # Constants can be added directly to NVL snippet.
    if isinstance(other, str) or isinstance(other, WeldObject):
        other_var = weld_obj.update(other)
        if isinstance(other, WeldObject):
            other_var = tmp.obj_id
            weld_obj.dependencies[other_var] = other
    else:
        other_var = "%s(%s)" % (ty_str, str(other))

    weld_template = """
       map(
         %(array)s,
         |a: %(ty)s| a %(op)s %(other)s
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "other": other_var,
                                          "op": op, "ty": ty_str}

    return weld_obj


def slice(array, start, size, ty):
    """
    Returns a new array-of-arrays with each array truncated, starting at
    index `start` for `length` characters.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
       map(
         %(array)s,
         |array: %(ty)s| slice(array, %(start)dL, %(size)dL)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "start": start,
                                          "ty": ty, "size": size}

    return weld_obj

def to_lower(array, ty):
    """
    Returns a new array of strings that are converted to lowercase

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
       map(
         %(array)s,
         |array: vec[i8]| map(array, |b:i8| if(b <= i8(90), b + i8(32), b))
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "ty": ty}

    return weld_obj

def contains(array, ty, string):
    """
    Checks if given string is contained in each string in the array.
    Output is a vec of booleans.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    string_obj = weld_obj.update(string)
    if isinstance(string, WeldObject):
        string_obj =  string.obj_id
        weld_obj.dependencies[string_obj] = string

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    (start, end) = 0, len(string)
    # Some odd bug where iterating on str and slicing str results
    # in a segfault
    weld_template = """
       map(
         %(array)s,
         |str: vec[%(ty)s]|
           let tstr = str;
           result(
             for(
               str,
               merger[i8,+](i8(0)),
               |b,i,e|
                 if(slice(tstr, i, i64(%(end)s)) == %(cmpstr)s,
                    merge(b, i8(1)),
                    b
                 )
              )
            ) > i8(0)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "ty": ty,
                                          "start": start, "end": end,
                                          "cmpstr": string_obj}

    return weld_obj


def count(array, ty):
    """
    Return number of non-NA/null observations in the Series
    TODO : Filter out NaN's once Weld supports NaN's

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
     len(%(array)s)
  """

    weld_obj.weld_code = weld_template % {"array": array_var}

    return weld_obj

def join(expr1, expr2, d1_keys, d2_keys, keys_type, d1_vals, df1_vals_ty, d2_vals, df2_vals_ty):
    """
       Computes a join on two tables
    """
    weld_obj = WeldObject(encoder_, decoder_)

    df1_var = weld_obj.update(expr1)
    if isinstance(expr1, WeldObject):
        df1_var = expr1.obj_id
        weld_obj.dependencies[df1_var] = expr1

    df2_var = weld_obj.update(expr2)
    if isinstance(expr2, WeldObject):
        df2_var = expr2.obj_id
        weld_obj.dependencies[df2_var] = expr2

    d1_key_fields = ", ".join(["e.$%s" % k for k in d1_keys])
    if len(d1_keys) > 1:
        d1_key_struct = "{%s}" % (d1_key_fields)
    else:
        d1_key_struct = d1_key_fields

    d1_val_fields = ", ".join(["e.$%s" % k for k in d1_vals])

    d2_key_fields = ", ".join(["e.$%s" % k for k in d2_keys])
    if len(d2_keys) > 1:
        d2_key_struct = "{%s}" % (d2_key_fields)
    else:
        d2_key_struct = d2_key_fields

    d2_val_fields = ", ".join(["e.$%s" % k for k in d2_vals])
    d2_val_fields2 = ", ".join(["e2.$%s" % i for i, k in enumerate(d2_vals)])
    d2_val_struct = "{%s}" % (d2_val_fields)

    weld_template = """
    let df2_join_table = result(
      for(
        %(df2)s,
        groupmerger[%(kty)s, %(df2ty)s],
        |b, i, e| merge(b, {%(df2key)s, %(df2vals)s})
      )
    );

    result(for(
      %(df1)s,
      appender,
      |b, i, e|
        for(
          lookup(df2_join_table, %(df1key)s),
          b,
          |b2, i2, e2| merge(b, {%(df1key)s, %(df1vals)s, %(df2vals2)s})
        )
    ))
    """

    weld_obj.weld_code = weld_template % {"df1":df1_var,
                                          "df1key":d1_key_struct,
                                          "df1vals": d1_val_fields,
                                          "df2":df2_var,
                                          "kty":keys_type,
                                          "df2ty":df2_vals_ty,
                                          "df2key":d2_key_struct,
                                          "df2vals":d2_val_struct,
                                          "df2vals2":d2_val_fields2}
    
    return weld_obj

def pivot_table(expr, value_index, value_ty, index_index, index_ty, columns_index, columns_ty, aggfunc):
    """
    Constructs a pivot table where the index_index and columns_index are used as keys and the value_index is used as the value which is aggregated.
    """

    weld_obj = WeldObject(encoder_, decoder_)
    zip_var = weld_obj.update(expr)
    if isinstance(expr, WeldObject):
        zip_var = expr.obj_id
        weld_obj.dependencies[zip_var] = expr

    if aggfunc == 'sum':
        weld_template = """
          let bs = for(
            %(zip_expr)s,
            {dictmerger[{%(ity)s,%(cty)s},%(vty)s,+], dictmerger[%(ity)s,i64,+], dictmerger[%(cty)s,i64,+]},
            |b, i, e| {merge(b.$0, {{e.$%(id)s, e.$%(cd)s}, e.$%(vd)s}),
                       merge(b.$1, {e.$%(id)s, 1L}),
                       merge(b.$2, {e.$%(cd)s, 1L})}
          );
          let agg_dict = result(bs.$0);
          let ind_vec = sort(map(tovec(result(bs.$1)), |x| x.$0), |x| x);
          let col_vec = map(tovec(result(bs.$2)), |x| x.$0);
          let pivot = map(
            col_vec,
            |x:%(cty)s|
            map(ind_vec, |y:%(ity)s| f64(lookup(agg_dict, {y, x})))
        );
        {ind_vec, pivot, col_vec}
    """
    elif aggfunc == 'mean':
        weld_template = """
          let bs = for(
            %(zip_expr)s,
            {dictmerger[{%(ity)s,%(cty)s},{%(vty)s, i64},+], dictmerger[%(ity)s,i64,+], dictmerger[%(cty)s,i64,+]},
            |b, i, e| {merge(b.$0, {{e.$%(id)s, e.$%(cd)s}, {e.$%(vd)s, 1L}}),
                       merge(b.$1, {e.$%(id)s, 1L}),
                       merge(b.$2, {e.$%(cd)s, 1L})}
          );
          let agg_dict = result(bs.$0);
          let ind_vec = sort(map(tovec(result(bs.$1)), |x| x.$0), |x| x);
          let col_vec = map(tovec(result(bs.$2)), |x| x.$0);
          let pivot = map(
            col_vec,
            |x:%(cty)s|
            map(ind_vec, |y:%(ity)s| (
              let sum_len_pair = lookup(agg_dict, {y, x});
              f64(sum_len_pair.$0) / f64(sum_len_pair.$1))
            )
        );
        {ind_vec, pivot, col_vec}
    """

    else:
        raise Exception("Aggregate operation %s not supported." % aggfunc)

    weld_obj.weld_code = weld_template % {"zip_expr": zip_var,
                                          "ity": index_ty,
                                          "cty": columns_ty,
                                          "vty": value_ty,
                                          "id": index_index,
                                          "cd": columns_index,
                                          "vd": value_index}
    return weld_obj

def get_pivot_column(pivot, column_name, column_type):
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_var = weld_obj.update(pivot)
    if isinstance(pivot, WeldObject):
        pivot_var = pivot.obj_id
        weld_obj.dependencies[pivot_var] = pivot

    column_name_var = weld_obj.update(column_name)
    if isinstance(column_name, WeldObject):
        column_name_var = column_name.obj_id
        weld_obj.dependencies[column_name_var] = column_name

    weld_template = """
      let col_dict = result(for(
        %(piv)s.$2,
        dictmerger[%(cty)s,i64,+],
        |b, i, e| merge(b, {e, i})
      ));
    {%(piv)s.$0, lookup(%(piv)s.$1, lookup(col_dict, %(colnm)s))}
    """

    weld_obj.weld_code = weld_template % {"piv":pivot_var,
                                          "cty":column_type,
                                          "colnm":column_name_var}
    return weld_obj

def pivot_sort(pivot, column_name, index_type, column_type, pivot_type):
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_var = weld_obj.update(pivot)
    if isinstance(pivot, WeldObject):
        pivot_var = pivot.obj_id
        weld_obj.dependencies[pivot_var] = pivot

    column_name_var = weld_obj.update(column_name)
    if isinstance(column_name, WeldObject):
        column_name_var = column_name.obj_id
        weld_obj.dependencies[column_name_var] = column_name

        # Note the key_column is hardcoded for the movielens workload
        # diff is located at the 2nd index.
    weld_template = """
      let key_col = lookup(%(piv)s.$1, 2L);
      let sorted_indices = map(
        sort(
          result(
            for(
              key_col,
              appender[{%(pvty)s,i64}],
              |b,i,e| merge(b, {e,i})
            )
          ),
          |x| x.$0
        ),
        |x| x.$1
      );
      let new_piv = map(
        %(piv)s.$1,
        |x| result(
          for(
            x,
            appender[%(pvty)s],
            |b,i,e| merge(b, lookup(x, lookup(sorted_indices, i)))
          )
        )
      );
      let new_index = result(
        for(
          %(piv)s.$0,
          appender[%(idty)s],
          |b,i,e| merge(b, lookup(%(piv)s.$0, lookup(sorted_indices, i)))
        )
      );
    {new_index, new_piv, %(piv)s.$2}
    """

    weld_obj.weld_code = weld_template % {"piv": pivot_var,
                                          "cty": column_type,
                                          "idty": index_type,
                                          "colnm": column_name_var,
                                          "pvty": pivot_type}
    return weld_obj

def set_pivot_column(pivot, column_name, item, pivot_type, column_type):
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_var = weld_obj.update(pivot)
    if isinstance(pivot, WeldObject):
        pivot_var = pivot.obj_id
        weld_obj.dependencies[pivot_var] = pivot

    column_name_var = weld_obj.update(column_name)
    if isinstance(column_name, WeldObject):
        column_name_var = column_name.obj_id
        weld_obj.dependencies[column_name_var] = column_name

    item_var = weld_obj.update(item)
    if isinstance(item, WeldObject):
        item_var = item.obj_id
        weld_obj.dependencies[item_var] = item

    weld_template = """
    let pv_cols = for(
      %(piv)s.$1,
      appender[%(pvty)s],
      |b, i, e| merge(b, e)
    );
    let col_names = for(
      %(piv)s.$2,
      appender[%(cty)s],
      |b, i, e| merge(b, e)
    );
    let new_pivot_cols = result(merge(pv_cols, %(item)s));
    let new_col_names = result(merge(col_names, %(colnm)s));

    {%(piv)s.$0, new_pivot_cols, new_col_names}
    """

    weld_obj.weld_code = weld_template % {"piv":pivot_var,
                                          "cty":column_type,
                                          "pvty":pivot_type,
                                          "item":item_var,
                                          "colnm":column_name_var}
    return weld_obj

def pivot_sum(pivot, value_type):
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_var = weld_obj.update(pivot)
    if isinstance(pivot, WeldObject):
        pivot_var = pivot.obj_id
        weld_obj.dependencies[pivot_var] = pivot

    weld_template = """
      result(for(
        %(piv)s.$0,
        appender[%(pty)s],
        |b1,i1,e1|
          merge(b1,
            result(for(
              %(piv)s.$1,
              merger[%(pty)s,+],
              |b2, i2, e2| merge(b2, lookup(e2, i1))
            ))
          )
       ))
    """

    weld_obj.weld_code = weld_template % {"piv":pivot_var,
                                         "pty":value_type}
    return weld_obj

def pivot_div(pivot, series, vec_type, val_type):
    weld_obj = WeldObject(encoder_, decoder_)

    pivot_var = weld_obj.update(pivot)
    if isinstance(pivot, WeldObject):
        pivot_var = pivot.obj_id
        weld_obj.dependencies[pivot_var] = pivot

    series_var = weld_obj.update(series)
    if isinstance(series, WeldObject):
        series_var = series.obj_id
        weld_obj.dependencies[series_var] = series

    weld_template = """
    {%(piv)s.$0,
      map(
        %(piv)s.$1,
        |v:%(vty)s|
          result(for(
            v,
            appender[f64],
            |b, i, e| merge(b, f64(e) / f64(lookup(%(srs)s, i)))
          ))
      ),
     %(piv)s.$2
    }
    """

    weld_obj.weld_code = weld_template % {"piv":pivot_var,
                                          "pty":val_type,
                                          "vty":vec_type,
                                          "srs":series_var}
    return weld_obj

def groupby_sum(columns, column_tys, grouping_columns, grouping_column_tys):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    if len(grouping_columns) == 1 and len(grouping_column_tys) == 1:
        grouping_column_var = weld_obj.update(grouping_columns[0])
        if isinstance(grouping_columns[0], WeldObject):
            grouping_column_var = grouping_columns[0].weld_code
        grouping_column_ty_str = "%s" % grouping_column_tys[0]
    else:
        grouping_column_vars = []
        for column in grouping_columns:
            column_var = weld_obj.update(column)
            if isinstance(column_var, WeldObject):
                column_var = column_var.weld_code
            grouping_column_vars.append(column_var)
        grouping_column_var = ", ".join(grouping_column_vars)
        grouping_column_tys = [str(ty) for ty in grouping_column_tys]
        grouping_column_ty_str = ", ".join(grouping_column_tys)
        grouping_column_ty_str = "{%s}" % grouping_column_ty_str

    columns_var_list = []
    for column in columns:
        column_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_var = column.obj_id
            weld_obj.dependencies[column_var] = column
        columns_var_list.append(column_var)

    if len(columns_var_list) == 1 and len(grouping_columns) == 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        result_str = "merge(b, e)"
    elif len(columns_var_list) == 1 and len(grouping_columns) > 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        key_str_list = []
        for i in xrange(0, len(grouping_columns)):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str = "e.$" + str(len(grouping_columns))
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)
    elif len(columns_var_list) > 1 and len(grouping_columns) == 1:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str = "e.$0"
        value_str_list = []
        for i in xrange(1, len(columns) + 1):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)
    else:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str_list = []
        key_size = len(grouping_columns)
        value_size = len(columns)
        for i in xrange(0, key_size):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str_list = []
        for i in xrange(key_size, key_size + value_size):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)
    weld_template = """
     tovec(
       result(
         for(
           zip(%(grouping_column)s, %(columns)s),
           dictmerger[%(gty)s, %(ty)s, +],
           |b, i, e| %(result)s
         )
       )
     )
  """

    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_var,
                                          "columns": columns_var,
                                          "result": result_str,
                                          "ty": tys_str,
                                          "gty": grouping_column_ty_str}
    return weld_obj

def groupby_std(columns, column_tys, grouping_columns, grouping_column_tys):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    if len(grouping_columns) == 1 and len(grouping_column_tys) == 1:
        grouping_column_var = weld_obj.update(grouping_columns[0])
        if isinstance(grouping_columns[0], WeldObject):
            grouping_column_var = grouping_columns[0].weld_code
        grouping_column_ty_str = "%s" % grouping_column_tys[0]
    else:
        grouping_column_vars = []
        for column in grouping_columns:
            column_var = weld_obj.update(column)
            if isinstance(column_var, WeldObject):
                column_var = column_var.weld_code
            grouping_column_vars.append(column_var)
        grouping_column_var = ", ".join(grouping_column_vars)
        grouping_column_tys = [str(ty) for ty in grouping_column_tys]
        grouping_column_ty_str = ", ".join(grouping_column_tys)
        grouping_column_ty_str = "{%s}" % grouping_column_ty_str

    columns_var_list = []
    for column in columns:
        column_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_var = column.obj_id
            weld_obj.dependencies[column_var] = column
        columns_var_list.append(column_var)

    if len(columns_var_list) == 1 and len(grouping_columns) == 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        result_str = "merge(b, e)"
        # TODO : The above change needs to apply for all result strings
        # These currently break at the moment
    elif len(columns_var_list) == 1 and len(grouping_columns) > 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        key_str_list = []
        for i in xrange(0, len(grouping_columns)):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str = "e.$" + str(len(grouping_columns))
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s, 1L})" % ", ".join(result_str_list)
    elif len(columns_var_list) > 1 and len(grouping_columns) == 1:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str = "e.$0"
        value_str_list = []
        for i in xrange(1, len(columns) + 1):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, %s)" % ", ".join(result_str_list)
    else:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str_list = []
        key_size = len(grouping_columns)
        value_size = len(columns)
        for i in xrange(0, key_size):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str_list = []
        for i in xrange(key_size, key_size + value_size):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, %s)" % ", ".join(result_str_list)
        # TODO Need to implement exponent operator
        # The mean dict's e.$1.$0 assumes this is a scalar vector and not a struct vector
        # Also pandas normalizes by n - 1
    weld_template = """
    let sum_dict = result(
      for(
        zip(%(grouping_column)s, %(columns)s),
        groupmerger[%(gty)s, %(ty)s],
        |b, i, e| %(result)s
      )
    );
    map(
        sort(tovec(sum_dict), |x| x.$0),
        |e| {e.$0, 
                   (let sum = result(for(e.$1, merger[%(ty)s,+], |b,i,a| merge(b, a)));
                    let m = f64(sum) / f64(len(e.$1)); 
                    let msqr = result(for(e.$1, merger[f64,+], |b,i,a| merge(b, (f64(a)-m)*(f64(a)-m))));
                    sqrt(msqr / f64(len(e.$1) - 1L)) 
                   )}
    )
  """


    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_var,
                                          "columns": columns_var,
                                          "result": result_str,
                                          "ty": tys_str,
                                          "gty": grouping_column_ty_str}
    return weld_obj

def groupby_size(columns, column_tys, grouping_columns, grouping_column_tys):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    if len(grouping_columns) == 1 and len(grouping_column_tys) == 1:
        grouping_column_var = weld_obj.update(grouping_columns[0])
        if isinstance(grouping_columns[0], WeldObject):
            grouping_column_var = grouping_columns[0].weld_code
        grouping_column_ty_str = "%s" % grouping_column_tys[0]
    else:
        grouping_column_vars = []
        for column in grouping_columns:
            column_var = weld_obj.update(column)
            if isinstance(column_var, WeldObject):
                column_var = column_var.weld_code
            grouping_column_vars.append(column_var)
        grouping_column_var = ", ".join(grouping_column_vars)
        grouping_column_var = "zip(%s)" % grouping_column_var
        grouping_column_tys = [str(ty) for ty in grouping_column_tys]
        grouping_column_ty_str = ", ".join(grouping_column_tys)
        grouping_column_ty_str = "{%s}" % grouping_column_ty_str

    weld_template = """
    let group = sort(
      tovec(
        result(
          for(
            %(grouping_column)s,
            dictmerger[%(gty)s, i64, +],
            |b, i, e| merge(b, {e, 1L})
          )
        )
      ),
      |x:{%(gty)s, i64}| x.$0
    );
    {
      map(
      group,
        |x:{%(gty)s,i64}| x.$0
      ),
      map(
        group,
        |x:{%(gty)s,i64}| x.$1
      )
    }
  """

    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_var,
                                          "gty": grouping_column_ty_str}
    return weld_obj

def groupby_sort(columns, column_tys, grouping_columns, grouping_column_tys, key_index, ascending):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    if len(grouping_columns) == 1 and len(grouping_column_tys) == 1:
        grouping_column_var = weld_obj.update(grouping_columns[0])
        if isinstance(grouping_columns[0], WeldObject):
            grouping_column_var = grouping_columns[0].weld_code
        grouping_column_ty_str = "%s" % grouping_column_tys[0]
    else:
        grouping_column_vars = []
        for column in grouping_columns:
            column_var = weld_obj.update(column)
            if isinstance(column, WeldObject):
                column_var = column.obj_id
                weld_obj.dependencies[column_var] = column
            grouping_column_vars.append(column_var)
        grouping_column_var = ", ".join(grouping_column_vars)
        grouping_column_tys = [str(ty) for ty in grouping_column_tys]
        grouping_column_ty_str = ", ".join(grouping_column_tys)
        grouping_column_ty_str = "{%s}" % grouping_column_ty_str

    columns_var_list = []
    for column in columns:
        column_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_var = column.obj_id
            weld_obj.dependencies[column_var] = column
        columns_var_list.append(column_var)

    if len(columns_var_list) == 1 and len(grouping_columns) == 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        result_str = "merge(b, e)"
    elif len(columns_var_list) == 1 and len(grouping_columns) > 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        key_str_list = []
        for i in xrange(0, len(grouping_columns)):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str = "e.$" + str(len(grouping_columns))
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)
    elif len(columns_var_list) > 1 and len(grouping_columns) == 1:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str = "e.$0"
        value_str_list = []
        for i in xrange(1, len(columns) + 1):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)
    else:
        columns_var = "%s" % ", ".join(columns_var_list)
        column_tys = [str(ty) for ty in column_tys]
        tys_str = "{%s}" % ", ".join(column_tys)
        key_str_list = []
        key_size = len(grouping_columns)
        value_size = len(columns)
        for i in xrange(0, key_size):
            key_str_list.append("e.$%d" % i)
        key_str = "{%s}" % ", ".join(key_str_list)
        value_str_list = []
        for i in xrange(key_size, key_size + value_size):
            value_str_list.append("e.$%d" % i)
        value_str = "{%s}" % ", ".join(value_str_list)
        result_str_list = [key_str, value_str]
        result_str = "merge(b, {%s})" % ", ".join(result_str_list)

    if key_index == None:
        key_str = "x"
    else :
        key_str = "x.$%d" % key_index

    if ascending == False:
        key_str = key_str + "* %s(-1)" % column_tys[key_index]

    weld_template = """
    @(grain_size:1)map(
     tovec(
       result(
         for(
           zip(%(grouping_column)s, %(columns)s),
           groupmerger[%(gty)s, %(ty)s],
           |b, i, e| %(result)s
         )
       )
     ),
    |a:{%(gty)s, vec[%(ty)s]}| {a.$0, sort(a.$1, |x:%(ty)s| %(key_str)s)}
    )
  """

    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_var,
                                          "columns": columns_var,
                                          "result": result_str,
                                          "ty": tys_str,
                                          "gty": grouping_column_ty_str,
                                          "key_str": key_str}
    return weld_obj

def flatten_group(expr, column_tys, grouping_column_tys):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    group_var = weld_obj.update(expr)

    num_group_cols = len(grouping_column_tys)
    if num_group_cols == 1:
        grouping_column_ty_str = "%s" % grouping_column_tys[0]
        grouping_column_key_str = "a.$0"
    else:
        grouping_column_tys = [str(ty) for ty in grouping_column_tys]
        grouping_column_ty_str = ", ".join(grouping_column_tys)
        grouping_column_ty_str = "{%s}" % grouping_column_ty_str
        grouping_column_keys = ["a.$0.$%s" % i for i in range(0, num_group_cols)]
        grouping_column_key_str = "{%s}" % ", ".join(grouping_column_keys)

    if len(column_tys) == 1:
        tys_str = "%s" % column_tys[0]
        column_values_str = "b.$1"
    else:
        column_tys = [str(ty) for ty in column_tys]

        tys_str = "{%s}" % ", ".join(column_tys)
        column_values = ["b.$%s" % i for i in range(0, len(column_tys))]
        column_values_str = "{%s}" % ", ".join(column_values)

    # TODO: The output in pandas keps the keys sorted (even if keys are structs)
    # We need to allow keyfunctions in sort by clauses to be able to compare structs.
    # sort(%(group_vec)s, |x:{%(gty)s, vec[%(ty)s]}| x.$0),
    weld_template = """
    flatten(
      map(
        %(group_vec)s,
        |a:{%(gty)s, vec[%(ty)s]}| map(a.$1, |b:%(ty)s| {%(gkeys)s, %(gvals)s})
      )
    )
  """

    weld_obj.weld_code = weld_template % {"group_vec": expr.weld_code,
                                          "gkeys": grouping_column_key_str,
                                          "gvals": column_values_str,
                                          "ty": tys_str,
                                          "gty": grouping_column_ty_str}
    return weld_obj

def grouped_slice(expr, type, start, size):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    weld_obj.update(expr)
    weld_template = """
       map(
           %(vec)s,
           |b: %(type)s| {b.$0, slice(b.$1, i64(%(start)s), i64(%(size)s))}
         )
    """

    weld_obj.weld_code = weld_template % {"vec": expr.weld_code,
                                          "type": type,
                                          "start": str(start),
                                          "size": str(size)}
    return weld_obj


def get_column(columns, column_tys, index):
    """
    Get column corresponding to passed-in index from ptr returned
    by groupBySum.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        index (int): index of selected column

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    columns_var = weld_obj.update(columns, tys=WeldVec(column_tys), override=False)
    if isinstance(columns, WeldObject):
        columns_var = columns.obj_id
        weld_obj.dependencies[columns_var] = columns

    weld_template = """
     map(
       %(columns)s,
       |elem: %(ty)s| elem.$%(index)s
     )
  """

    weld_obj.weld_code = weld_template % {"columns": columns_var,
                                          "ty": column_tys,
                                          "index": index}
    return weld_obj
