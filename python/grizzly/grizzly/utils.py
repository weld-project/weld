import grizzly_impl

from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *

from seriesweld import *
from dataframeweld import *

def merge(df1, df2):
    if isinstance(df1, DataFrameWeld):
        # TODO need to passin filtering by predicates as well here
        tys = []
        for col_name, raw_column in df1.raw_columns.items():
            dtype = str(raw_column.dtype)
            if dtype == 'object' or dtype == '|S64':
                weld_type = WeldVec(WeldChar())
            else:
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            tys.append(weld_type)

        if len(tys) == 1:
            weld_type = tys[0]
        else:
            weld_type = WeldStruct(tys)

        df1 = DataFrameWeldExpr(
            grizzly_impl.zip_columns(
                df1.raw_columns.values()
            ),
            df1.raw_columns.keys(),
            weld_type
        )
    if isinstance(df2, DataFrameWeld):
        tys = []
        for col_name, raw_column in df2.raw_columns.items():
            dtype = str(raw_column.dtype)
            if dtype == 'object' or dtype == '|S64':
                weld_type = WeldVec(WeldChar())
            else:
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            tys.append(weld_type)

        if len(tys) == 1:
            weld_type = tys[0]
        else:
            weld_type = WeldStruct(tys)
        df2 = DataFrameWeldExpr(
            grizzly_impl.zip_columns(
                df2.raw_columns.values()
            ),
            df2.raw_columns.keys(),
            weld_type
        )
    return df1.merge(df2)

def group_eval(objs, passes=None):
    LazyOpResults = []
    for ob in objs:
        if isinstance(ob, SeriesWeld):
            if ob.index_type is not None:
                weld_type = WeldStruct([WeldVec(ob.index_type), WeldVec(ob.weld_type)])
                LazyOpResults.append(LazyOpResult(ob.expr, weld_type, 0))
        else:
            LazyOpResults.append(LazyOpResult(ob.expr, ob.weld_type, 0))
    
    results = group(LazyOpResults).evaluate((True, -1), passes=passes)
    pd_results = []
    for i, result in enumerate(results):
        ob = objs[i]
        if isinstance(ob, SeriesWeld):
            if ob.index_type is not None:
                index, column = result
                series = pd.Series(column, index)
                series.index.rename(ob.index_name, True)
                pd_results.append(series)
            else:
                pd_results.append(series)
        if isinstance(ob, DataFrameWeldExpr):
            if ob.is_pivot:
                index, pivot, columns = result
                df_dict = {}
                for i, column_name in enumerate(columns):
                    df_dict[column_name] = pivot[i]
                pd_results.append(pd.DataFrame(df_dict, index=index))
            else:
                columns = result
                df_dict = {}
                for i, column_name in enumerate(ob.column_names):
                    df_dict[column_name] = columns[i]
                pd_results.append(pd.DataFrame(df_dict))
    return pd_results

def group(exprs):
    weld_type = [to_weld_type(expr.weld_type, expr.dim) for expr in exprs]
    exprs = [expr.expr for expr in exprs]
    weld_obj = WeldObject(grizzly_impl.encoder_, grizzly_impl.decoder_)
    weld_type = WeldStruct(weld_type)
    dim = 0

    expr_names = [expr.obj_id for expr in exprs]
    for expr in exprs:
        weld_obj.update(expr)
    weld_obj.weld_code = "{%s}" % ", ".join(expr_names)
    for expr in exprs:
        weld_obj.dependencies[expr.obj_id] = expr

    return LazyOpResult(weld_obj, weld_type, dim)
