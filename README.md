# Weld

[![Build Status](https://travis-ci.org/weld-project/weld.svg?branch=master)](https://travis-ci.org/weld-project/weld)

Weld is a language and runtime for improving the performance of data-intensive applications. It optimizes across libraries and functions by expressing the core computations in libraries using a common intermediate representation, and optimizing across each framework.

Modern analytics applications combine multiple functions from different libraries and frameworks to build complex workflows. Even though individual functions can achieve high performance in isolation, the performance of the combined workflow is often an order of magnitude below hardware limits due to extensive data movement across the functions. Weld’s take on solving this problem is to lazily build up a computation for the entire workflow, and then optimizing and evaluating it only when a result is needed.

You can join the discussion on Weld on our [Google Group](https://groups.google.com/forum/#!forum/weld-users) or post on the Weld mailing list at [weld-group@lists.stanford.edu](mailto:weld-group@lists.stanford.edu).

## Contents

  * [Building](#building)
      - [MacOS LLVM Installation](#macos-llvm-installation)
      - [Ubuntu LLVM Installation](#ubuntu-llvm-installation)
      - [Building Weld](#building-weld)
  * [Documentation](#documentation)
  * [Grizzly (Pandas on Weld)](#grizzly)
  * [Tools](#tools)

## Building

To build Weld, you need [Rust 1.24 or higher](http://rust-lang.org) and [LLVM](http://llvm.org) 3.8 or newer.

To install Rust, follow the steps [here](https://rustup.rs). You can verify that Rust was installed correctly on your system by typing `rustc` into your shell.

#### MacOS LLVM Installation

To install LLVM on macOS, first install [Homebrew](https://brew.sh/). Then:

```bash
$ brew install llvm@3.8
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up (note that you might need to add `sudo` at the start of this command):

```bash
$ ln -s /usr/local/bin/llvm-config-3.8 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `3.8.x` or newer.

Enter the `weld_rt/cpp` directory and try running `make`. If the command fails with errors related to missing header files, you may need to install XCode and/or XCode Command Line Tools. Run `xcode-select --install` to do this.

#### Ubuntu LLVM Installation

To install LLVM on Ubuntu :

```bash
$ sudo apt-get install llvm-3.8
$ sudo apt-get install llvm-3.8-dev
$ sudo apt-get install clang-3.8
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up:

```bash
$ ln -s /usr/bin/llvm-config-3.8 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `3.8.x` or newer.

#### Building Weld

With LLVM and Rust installed, you can build Weld. Clone this repository, set the `WELD_HOME` environment variable, and build using `cargo`:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld/
$ export WELD_HOME=`pwd`
$ cargo build --release
```

**Note:** If you are using a version of LLVM newer than 3.8, you will have to change the `llvm-sys` crate dependency in `easy_ll/Cargo.toml` to match (e.g. `40.0.0` for LLVM 4.0.0). You may also need to create additional symlinks for some packages that omit the version suffix when installing the latest version, e.g. for LLVM 4.0:

```bash
$ ln -s /usr/local/bin/clang /usr/local/bin/clang-4.0
$ ln -s /usr/local/bin/llvm-link /usr/local/bin/llvm-link-4.0
```

Weld builds two dynamically linked libraries (`.so` files on Linux and `.dylib` files on Mac): `libweld` and `libweldrt`.

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

## Python Bindings

Weld's Python bindings are in [`python/weld`](https://github.com/weld-project/weld/tree/master/python/weld), with examples in [`examples/python`](https://github.com/weld-project/weld/tree/master/examples/python).

## Grizzly

**Grizzly** is a subset of [Pandas](http://pandas.pydata.org/) integrated with Weld. Details on how to use Grizzly are in
[`python/grizzly`](https://github.com/weld-project/weld/tree/master/python/grizzly).
Some example workloads that make use of Grizzly are in [`examples/python/grizzly`](https://github.com/weld-project/weld/tree/master/examples/python/grizzly).
To run Grizzly, you will also need the `WELD_HOME` environment variable to be set, because Grizzly needs to find its own native library through this variable.

## Testing

`cargo test` runs unit and integration tests. A test name substring filter can be used to run a subset of the tests:

   ```
   cargo test <substring to match in test name>
   ```

## Tools

This repository contains a number of useful command line tools which are built
automatically with the main Weld repository, including an interactive REPL for
inspecting and debugging programs.  More information on those tools can be
found under [docs/tools.md](https://github.com/weld-project/weld/tree/master/docs/tools.md).

## Current performance metrics

![Current performance metrics](https://s3-us-west-2.amazonaws.com/weld-travis-ci/master/performance.png)

