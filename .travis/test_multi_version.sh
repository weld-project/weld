#!/bin/bash
#
# A script to test Weld on various versions of Python and LLVM.

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3

# Weld Core Tests
# ----------------------------------------------------------

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" Cargo.toml

# Python Tests
# ----------------------------------------------------------

# Install Python Requirements
rustup default nightly
pushd weld-python/
pip install -e .

# Run the tests
pytest

popd
