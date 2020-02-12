#!/bin/bash

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
export WELD_HOME=`pwd`

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" Cargo.toml

# build and test
cargo clippy
cargo fmt -- --check
cargo build #--release
cargo test

# Python Tests
cd $WELD_HOME
virtualenv travis-test-env
source travis-test-env/bin/activate
cd weld-python
pip install -r requirements.txt

# Install Rust Requirements
rustup toolchain install nightly
rustup default nightly
pip install -e .

# Run the tests
pytest
