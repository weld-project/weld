"""
String functions exported as weldfuncs.

Each function takes an argument representing an array of strings, and outputs a
program that applies some transformation on each string. The functions are annotated
with `weld.lazy.weldfunc`, so they accept `WeldLazy` objects and return functions
for constructing Weld programs.

"""

import weld.lazy

def string_to_weld_literal(s):
    """
    Converts a string to a UTF-8 encoded Weld literal byte-vector.

    Examples
    --------
    >>> string_to_weld_literal('hello')
    '[104c,101c,108c,108c,111c]'

    """
    return "[" + ",".join([str(b) + 'c' for b in list(s.encode('utf-8'))]) + "]"

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
        slice(e, start_i - 1L, end_i - start_i + 3L)
    )""".format(stringarr=stringarr, is_whitespace=is_whitespace)

@weld.lazy.weldfunc
def contains(stringarr, pat):
    """
    Check whether each element contains the substring 'pat', and returns
    a boolean array of the results.

    For now, 'pat' must be a string literal.
    """
    define_pat = "let pat = {};".format(string_to_weld_literal(pat))

    return """
    {define_pat}
    let lenPat = len(pat);
        map({stringarr},
            |e: vec[i8]|
                let lenString = len(e);
                if(lenPat > lenString,
                    false,
                    # start by assuming pat is not found, until proven it is
                    let words_iter_res = iterate({{0L, false}},
                        |p|
                            let e_i = p.$0;
                            let pat_i = 0L;
                            # start by assuming the substring and pat are the same, until proven otherwise
                            let word_check_res = iterate({{e_i, pat_i, true}},
                                |q|
                                    let found = lookup(e, q.$0) == lookup(pat, q.$1);
                                    {{
                                        {{q.$0 + 1L, q.$1 + 1L, found}},
                                        q.$1 + 1L < lenPat &&
                                        found == true
                                    }}
                            ).$2;
                            {{
                                {{p.$0 + 1L, word_check_res}},
                                p.$0 + lenPat < lenString &&
                                word_check_res == false
                            }}
                    ).$1;
                    words_iter_res
                )
        )""".format(stringarr=stringarr, define_pat=define_pat)
