"""
String functions exported as weldfuncs.

Each function takes an argument representing an array of strings, and outputs a
program that applies some transformation on each string. The functions are annotated
with `weld.lazy.weldfunc`, so they accept `WeldLazy` objects and return functions
for constructing Weld programs.

These are adapted from
https://github.com/weld-project/baloo/blob/master/baloo/weld/weld_str.py.

We may choose to re-implement these as UDF calls to Rust's UTF-8 library in the future.

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

@weld.lazy.weldfunc
def startswith(stringarr, pat):
    """
    Check whether each element starts with the substring 'pat', and returns
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
                iterate({{0L, true}},
                    |q|
                        let found = lookup(e, q.$0) == lookup(pat, q.$0);
                        {{
                            {{q.$0 + 1L, found}},
                            q.$0 + 1L < lenPat &&
                            found == true
                        }}
                ).$1
            )
    )""".format(stringarr=stringarr, define_pat=define_pat)


@weld.lazy.weldfunc
def endswith(stringarr, pat):
    """
    Check whether each element ends with the substring 'pat', and returns
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
                iterate({{lenString - lenPat, 0L, true}},
                    |q|
                        let found = lookup(e, q.$0) == lookup(pat, q.$1);
                        {{
                            {{q.$0 + 1L, q.$1 + 1L, found}},
                            q.$1 + 1L < lenPat &&
                            found == true
                        }}
                ).$2
            )
    )""".format(stringarr=stringarr, define_pat=define_pat)

@weld.lazy.weldfunc
def find(stringarr, sub, start, end=None):
    """
    Searches for 'sub' in each string in the range 'start', 'end'.  Returns a
    i64 array with -1 for unfound strings, or the index of the found string.

    'sub' must be a Python string. 'start' and 'end' must be integers.

    """

    start = "i64({})".format(start)
    if end is None:
        end = 'len(e)'
    else:
        end = "i64({})".format(end)

    define_sub = "let sub = {};".format(string_to_weld_literal(sub))

    return """
    {define_sub}
    let lenSub = len(sub);
    map({stringarr},
        |e: vec[i8]|
            let start = {start};
            let size = {end} - start;
            if (start < 0L,
                -1L,
                let string = slice(e, start, size);
                let lenString = len(string);
                if(lenSub > lenString,
                    -1L,
                    # start by assuming sub is not found, until proven it is
                    let words_iter_res = iterate({{0L, false}},
                        |p|
                            let e_i = p.$0;
                            let pat_i = 0L;
                            # start by assuming the substring and sub are the same, until proven otherwise
                            let word_check_res = iterate({{e_i, pat_i, true}},
                                |q|
                                    let found = lookup(string, q.$0) == lookup(sub, q.$1);
                                    {{
                                        {{q.$0 + 1L, q.$1 + 1L, found}},
                                        q.$1 + 1L < lenSub &&
                                        found == true
                                    }}
                            ).$2;
                            {{
                                {{p.$0 + 1L, word_check_res}},
                                p.$0 + lenSub < lenString &&
                                word_check_res == false
                            }}
                    );
                    if(words_iter_res.$1 == true,
                        words_iter_res.$0 - 1L + start,
                        -1L
                    )
                )
            )
    )""".format(stringarr=stringarr, define_sub=define_sub, start=start, end=end)


@weld.lazy.weldfunc
def replace(stringarr, pat, rep):
    """
    Replace the first occurrence iof 'pat' in each string with with 'rep'.

    For now, 'pat' and 'rep' must be Python strings.

    """
    define_pat = "let pat = {};".format(string_to_weld_literal(pat))
    define_rep = "let rep = {};".format(string_to_weld_literal(rep))

    return  """
    {define_pat}
    {define_rep}
    let lenPat = len(pat);
    map({stringarr},
        |e: vec[i8]|
            let lenString = len(e);
            if(lenPat > lenString,
                e,
                # start by assuming sub is not found, until proven it is
                let words_iter_res = iterate({{0L, false}},
                    |p|
                        let e_i = p.$0;
                        let pat_i = 0L;
                        # start by assuming the substring and sub are the same, until proven otherwise
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
                );
                if(words_iter_res.$1 == true,
                    let rep_from = words_iter_res.$0 - 1L;
                    let rep_to = rep_from + lenPat;
                    let res = appender[i8];
                    let res = for(slice(e, 0L, rep_from),
                        res,
                        |c: appender[i8], j: i64, f: i8|
                            merge(c, f)
                    );
                    let res = for(rep,
                        res,
                        |c: appender[i8], j: i64, f: i8|
                            merge(c, f)
                    );
                    let res = for(slice(e, rep_to, lenString),
                        res,
                        |c: appender[i8], j: i64, f: i8|
                            merge(c, f)
                    );
                    result(res),
                    e
                )
            )
    )""".format(stringarr=stringarr, define_pat=define_pat, define_rep=define_rep)
