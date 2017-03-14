# Weld

Weld is a language and runtime for improving the performance of data-intensive applications. It optimizes across libraries and functions by expressing the core computations in libraries using a common intermediate representation, and optimizing across each framework.

Modern analytics applications combine multiple functions from different libraries and frameworks to build complex workflows. Even though individual functions can achieve high performance in isolation, the performance of the combined workflow is often an order of magnitude below hardware limits due to extensive data movement across the functions. Weld’s take on solving this problem is to lazily build up a computation for the entire workflow, and then optimizing and evaluating it only when a result is needed.

## Contents

  * [Building](#building)
      - [MacOS LLVM Installation](#macos-llvm-installation)
      - [Ubuntu LLVM Installation](#ubuntu-llvm-installation)
      - [Building Weld](#building-weld)
  * [Documentation](#documentation)
  * [Grizzly](#grizzly)
  * [Running an Interactive REPL](#running-an-interactive-repl)
  * [Benchmarking](#benchmarking)

## Building

To build Weld, you need [Rust 1.13 or higher](http://rust-lang.org) and [LLVM](http://llvm.org) 3.8.

To install Rust, follow the steps [here](https://rustup.rs). You can verify that Rust was installed correctly on your system by typing `rustc` into your shell.

#### MacOS LLVM Installation

To install LLVM on macOS, first install [brew](https://brew.sh/). Then:

```bash
$ brew install llvm38
$ export PATH=$PATH:/usr/local/bin
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up:

```bash
$ ln -s /usr/local/bin/llvm-config-3.8 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `3.8.x`.

#### Ubuntu LLVM Installation

To install LLVM on Ubuntu :

```bash
$ sudo apt install llvm-3.8
$ sudo apt install clang-3.8
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up:

```bash
$ ln -s /usr/bin/llvm-config-3.8 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `3.8.x`.

#### Building Weld

With LLVM and Rust installed, you can build Weld. Clone this repository, set the `WELD_HOME` environment variable, and build using `cargo`:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld/
$ export WELD_HOME=`pwd`
$ cargo build --release
```

Weld builds two dynamically linked libraries (`.so` files on Linux and `.dylib` files on macOS): `libweld` and `libweldrt`. Both of these libraries must be on the `LD_LIBRARY_PATH`. By default, the libraries are in `$WELD_HOME/target/release` and `$WELD_HOME/weld_rt/target/release`. Set up the `LD_LIBRARY_PATH` as follows:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WELD_HOME/weld_rt/target/release:$WELD_HOME/target/release
```

Finally, run the unit and integration tests:

```bash
$ cargo test
```

## Documentation

The `docs/` directory contains documentation for the different components of Weld.

* [language.md](https://github.com/weld-project/weld/blob/master/docs/language.md) describes the syntax of the Weld IR.
* [api.md](https://github.com/weld-project/weld/blob/master/docs/api.md) describes the low-level C API for interfacing with Weld.
* [python.md](https://github.com/weld-project/weld/blob/master/docs/python.md) gives an overview of the Python API.
* [tutorial.md](https://github.com/weld-project/weld/blob/master/docs/tutorial.md) contains a tutorial for how to build a small vector library using Weld.

## Grizzly

**Grizzly** is a port of the [Pandas](pandas.pydata.org/) framework. Details on how to use Grizzly are under `python/grizzly`. 

## Running an Interactive REPL

* `cargo test` runs unit and integration tests. A test name substring filter can be used to run a subset of the tests:
   
   ```
   cargo test <substring to match in test name>
   ```

* The `target/release/repl` program is a simple "shell" where one can type Weld programs and see
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

## Benchmarking

`cargo bench` runs benchmarks under the `benches/` directory. The results of the benchmarks are written to a file called `benches.csv`. To specify specific benchmarks to run:

```
$ cargo bench [benchmark-name]
```

If a benchmark name is not provided, all benchmarks are run.

