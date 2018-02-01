# Strings in Weld

Weld has limited support for working with string constants.
String literals (delimited by quotation marks `"`) can be passed to Weld functions and manipulated, for example, via `CUDF` functions.
String manipulation operations are not currently supported natively in Weld.

Unlike Rust, which supports UTF-8, Weld strings are restricted to valid ASCII.