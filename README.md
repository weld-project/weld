# Weld

## Building

To build Weld, you need [Rust stable](http://rust-lang.org) and [LLVM](http://llvm.org) 3.6 or
higher. Set `PATH` so that `llvm-config` from your installation of LLVM is on the path and then
run `cargo build` in the root directory.

Note that the LLVM version included by default on Mac OS X is older than 3.6, so you will need
to update `PATH` temporarily while building Weld. You can check the version with
`llvm-config --version`.

## Testing

* `cargo test` runs unit and integration tests.
* The `target/debug/repl` program is a simple "shell" where one can type Weld programs and see
  the results of parsing, macro substitution and type inference.

## Modules

Weld is split up into several modules, partly to speed up compilation. They are as follows:
* `weld_error`: common error type used throughout Weld (we might grow this to be a general
  utilities module).
* `weld_ast`: data types for abstract syntax trees.
* `weld_parser`: grammar and parser using [lalrpop](https://github.com/nikomatsakis/lalrpop).
* `weld_transform`: various transformations including type inference (from partially typed to
  fully typed ASTs), macro substitution, and others.
* `weld_llvm`: code generator for LLVM.
* `weld`: main module that contains integration tests and will eventually contain a public API.