#!/bin/bash

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" easy_ll/Cargo.toml

# build and test
cargo clean
cargo build --release
cargo test
python python/grizzly/tests/grizzly_test.py
python python/grizzly/tests/numpy_weld_test.py
