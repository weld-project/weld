#!/bin/bash
#
# A script to test Weld on various versions of Python and LLVM.

LLVM_VERSION=$1
PYTHON_VERSION=$2

export LLVM_SYS_60_PREFIX=$(llvm-config-$LLVM_VERSION --libdir)

# Weld Core Tests
# ----------------------------------------------------------

# build and test
cargo clippy
cargo fmt -- --check
# Make sure the release build works.
cargo build --release
# Test uses the debug build.
cargo test

# Python Tests
# ----------------------------------------------------------

# Install Python Requirements
rustup default nightly
pushd weld-python/
pip install -e .

# Run the tests
pytest

popd
