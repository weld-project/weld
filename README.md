# Weld

## Building

To build Weld, you need [Rust 1.13 or higher](http://rust-lang.org) and [LLVM](http://llvm.org) 3.6 or
higher. Set `PATH` so that `llvm-config` from your installation of LLVM is on the path and then
run `cargo build` in the root directory.

Note that the LLVM version included by default on Mac OS X is older than 3.6, so you will need
to update `PATH` temporarily while building Weld. You can check the version with
`llvm-config --version`.

## Testing

* `cargo test` runs unit and integration tests.
* The `target/debug/repl` program is a simple "shell" where one can type Weld programs and see
  the results of parsing, macro substitution and type inference.

Example `repl` session:
```
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
