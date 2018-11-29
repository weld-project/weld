# Weld

[![Build Status](https://travis-ci.org/weld-project/weld.svg?branch=master)](https://travis-ci.org/weld-project/weld)

**This is an under-development branch that only supports single-threaded execution and [lacks a few features from the main branch](#llvm-st-unsupported-features). For multi-threaded support, use the `master` branch. For workloads that only require single-threaded execution and do not rely on the missing features, this branch provides a next-generation backend that should deliver substantially improved performance.**

### LLVM ST Unsupported Features
This backend currently lacks a few features. These features are tracked via the `unimplemented.sh` script.

* The `NdIter` iterator
* Vector keys in dictionaries where the vector contain pointers.

--- 

[Documentation](https://www.weld.rs/docs/weld/)

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

To build Weld, you need [the latest stable version of Rust](http://rust-lang.org) and [LLVM](http://llvm.org) 6.0 or newer.

To install Rust, follow the steps [here](https://rustup.rs). You can verify that Rust was installed correctly on your system by typing `rustc` into your shell. If you already have Rust and  `rustup` installed, you can upgrade to the latest stable version with:

```bash
$ rustup update stable
```

#### MacOS LLVM Installation

To install LLVM on macOS, first install [Homebrew](https://brew.sh/). Then:

```bash
$ brew install llvm@6
```

Weld's dependencies require `llvm-config` on `$PATH`, so you may need to create a symbolic link so the correct `llvm-config` is picked up (note that you might need to add `sudo` at the start of this command):

```bash
$ ln -s /usr/local/Cellar/llvm/6.0.0/bin/llvm-config /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `6.0.x`.

Enter the `weld_rt/cpp` directory and try running `make`. If the command fails with errors related to missing header files, you may need to install XCode and/or XCode Command Line Tools. Run `xcode-select --install` to do this.

#### Ubuntu LLVM Installation

To install LLVM on Ubuntu, get the LLVM 6.0 sources and then `apt-get`:

On Ubuntu 16.04 (Xenial):
```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main"
sudo apt-get update
sudo apt-get install llvm-6.0-dev clang-6.0
```
On Ubuntu 14.04 (Trusty):
```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main"

# gcc backport is required on 14.04, for libstdc++. See https://apt.llvm.org/
sudo apt-add-repository "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main"
sudo apt-get update
sudo apt-get install llvm-6.0-dev clang-6.0
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up. `sudo` may be required:

```bash
$ ln -s /usr/bin/llvm-config-6.0 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `6.0.x` or newer.

You will also need `zlib`:

```bash
$ sudo apt-get install zlib1g-dev
```

#### Building Weld

With LLVM and Rust installed, you can build Weld. Clone this repository, set the `WELD_HOME` environment variable, and build using `cargo`:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld/
$ export WELD_HOME=`pwd`
$ cargo build --release
```
Weld builds two dynamically linked libraries (`.so` files on Linux and `.dylib` files on Mac): `libweld` and `libweldrt`.

Finally, run the unit and integration tests:

```bash
$ cargo test
```

## Documentation

The [Rust Weld crate](https://crates.io/crates/weld) is documented [here](https://www.weld.rs/docs/weld/).

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
