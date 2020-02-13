#!/bin/bash
#
# A script to test Weld on various versions of Python and LLVM.

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3

# Weld Core Tests
# ----------------------------------------------------------

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
export WELD_HOME=`pwd`

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" Cargo.toml

# Python Tests
# ----------------------------------------------------------

cd $WELD_HOME
python$PYTHON_VERSION -m venv travis-test-env
source travis-test-env/bin/activate
cd weld-python

pip install numpy pandas pytest setuptools-rust

# Install Rust Requirements
rustup toolchain install nightly
rustup default nightly
pip install -e .

# Run the tests
pytest

deactivate
cd $WELD_HOME
