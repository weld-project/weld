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
def capitalize(stringarr):
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


@weld.lazy.weldfunc
def get(stringarr, i):
    """
    Retrieves the character at index 'i'.

    If 'i' is greater than the string length, returns '\0'.

    """
    i = "i64({})".format(i)
    return """map(
    {stringarr},
    |e: vec[i8]|
        let lenString = len(e);
        if({i} >= lenString,
            [0c],
            if({i} > 0L,
                result(merge(appender[i8], lookup(slice(e, 0L, lenString), {i}))),
                if ({i} > -lenString,
                    result(merge(appender[i8], lookup(slice(e, lenString, {i}), {i}))),
                    [0c]
                )
            )
        )
    )""".format(stringarr=stringarr, i=i)


@weld.lazy.weldfunc
def strip(stringarr):
    """
    Strip whitespace from the start of each string.

    """
    # From https://en.wikipedia.org/wiki/Whitespace_character.
    is_whitespace = "((lookup(e, p) == 32c) || (lookup(e, p)  >= 9c && lookup(e, p) <= 13c))"
    # +3L = +1 compensate start_i already +1'ed, +1 compensate end_i already -1'ed, +1 compensate for slice with size
    return """map(
    {stringarr},
    |e: vec[i8]|
        let lenString = len(e);
        let start_i = iterate(0L, |p| {{p + 1L, p < lenString && {is_whitespace}}});
        let end_i = iterate(lenString - 1L, |p| {{p - 1L, p > start_i && {is_whitespace}}});
        # slice(e, start_i - 1L, lenString - start_i + 1L)
        slice(e, start_i - 1L, end_i - start_i + 3L)
    )""".format(stringarr=stringarr, is_whitespace=is_whitespace)
