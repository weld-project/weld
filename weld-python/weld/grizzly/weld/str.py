"""
String functions exported as weldfuncs.

Each function takes an argument representing an array of strings, and outputs a
program that applies some transformation on each string. The functions are annotated
with `weld.lazy.weldfunc`, so they accept `WeldLazy` objects and return functions
for constructing Weld programs.

"""

import weld.lazy

@weld.lazy.weldfunc
def lower(stringarr):
    """
    Convert values to lowercase.

    """
    return """map(
    {stringarr},
    |e: vec[i8]|
        result(
            for(e,
                appender[i8],
                |c: appender[i8], j: i64, f: i8|
                    if(f > 64c && f < 91c,
                        merge(c, f + 32c),
                        merge(c, f))
            )
        )
    )""".format(stringarr=stringarr)


@weld.lazy.weldfunc
def upper(stringarr):
    """
    Convert values to uppercase.

    """
    return """map(
    {stringarr},
    |e: vec[i8]|
        result(
            for(e,
                appender[i8],
                |c: appender[i8], j: i64, f: i8|
                    if(f > 96c && f < 123c,
                        merge(c, f - 32c),
                        merge(c, f))
            )
        )
    )""".format(stringarr=stringarr)

@weld.lazy.weldfunc
def weld_str_capitalize(stringarr):
    """
    Capitalize first letter.

    """
    return """map(
    {stringarr},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenString > 0L,
            let res = appender[i8];
            let firstChar = lookup(e, 0L);
            let res = if(firstChar > 96c && firstChar < 123c, merge(res, firstChar - 32c), merge(res, firstChar));
            result(
                for(slice(e, 1L, lenString - 1L),
                    res,
                    |c: appender[i8], j: i64, f: i8|
                        if(f > 64c && f < 91c,
                            merge(c, f + 32c),
                            merge(c, f)
                        )
                )
            ),
            e)
    )""".format(stringarr=stringarr)
