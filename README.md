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
