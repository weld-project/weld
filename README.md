# Weld

## Building

To build Weld, you need [Rust 1.13 or higher](http://rust-lang.org) and [LLVM](http://llvm.org) 3.8 or
higher.

To install Rust, follow the steps [here](https://rustup.rs). You can verify that Rust was installed correctly on your system by typing `rustc` into your shell.

### MacOS Installation

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

With LLVM and Rust installed, you can build Weld. Clone this repository and build using `cargo`:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld/
$ cargo build
```

Set the `WELD_HOME` environment variable and run tests:

```bash
$ export WELD_HOME=/path/to/weld/directory
$ cargo test
```

### Ubuntu Installation

To install LLVM on ubuntu :

```bash
$ sudo apt install llvm-3.8
$ export PATH=$PATH:/usr/local/bin
```

Weld's dependencies require `llvm-config`, so you may need to create a symbolic link so the correct `llvm-config` is picked up:

```bash
$ ln -s /usr/bin/llvm-config-3.8 /usr/local/bin/llvm-config
```

To make sure this worked correctly, run `llvm-config --version`. You should see `3.8.x`.

With LLVM and Rust installed, you can build Weld. Clone this repository and build using `cargo`:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld/
$ cargo build
$ cargo test
```

Set the `WELD_HOME` environment variable and run tests:

```bash
$ export WELD_HOME=/path/to/weld/directory
$ cargo test
```

## Running an Interactive REPL

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
