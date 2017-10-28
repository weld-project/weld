
The Weld repository contains a few command line tools to facilitate development.


* [REPL](#repl)
* [Header Generation](#header-generation)

## REPL

The `target/release/repl` program is a simple "shell" where one can type Weld programs and see the results of parsing, macro substitution and type inference.

Example `repl` session:
```
$ ./target/debug/repl
> let a = 5 + 2; a + a
Raw structure: [...]

After macro substitution:
let a=((5+2));(a+a)

After inlining applies:
let a=((5+2));(a+a)

After type inference:
let a:i32=((5+2));(a:i32+a:i32)

Expression type: i32

> map([1, 2], |x| x+1)
Raw structure: [...]

After macro substitution:
result(for([1,2],appender[?],|b,x|merge(b,(|x|(x+1))(x))))

After inlining applies:
result(for([1,2],appender[?],|b,x|merge(b,(x+1))))

After type inference:
result(for([1,2],appender[i32],|b:appender[i32],x:i32|merge(b:appender[i32],(x:i32+1))))

Expression type: vec[i32]
```

Passing a `Lambda` expression will also perform LLVM code generation - other expression types will only perform parsing, type inference, and IR transformations.

The REPL tool can also take a number of options (e.g., to compile a Weld program into LLVM or set the logging level).
Run `target/release/repl --help` to see the available options.

## Header Generation

The `target/release/hdrgen` program takes a Weld program and generates a C++ header file, containing the argument and return types for the Weld program. Example:

```bash
$ cat program.weld
|s: {i32,i32,f32}| [s, s, s]
$ target/release/hdrgen -i program.weld
#ifndef _WELD_CPP_HEADER_
#define _WELD_CPP_HEADER_

// THIS IS A GENERATED C++ FILE.

#include <stdint.h> // For explicitly sized integer types.
#include <stdlib.h> // For malloc

// Defines Weld's primitive numeric types.
typedef bool        i1;

<some more generated code>

#endif /* _WELD_CPP_HEADER_ */
```
