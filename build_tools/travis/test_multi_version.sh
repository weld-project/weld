#!/bin/bash

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3
VENV_HOME=`pwd`/.virtualenv

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
export WELD_HOME=`pwd`

source $VENV_HOME/python$PYTHON_VERSION/bin/activate
cd python/pyweld
python setup.py install
cd ../..

cd python/grizzly
python setup.py install
cd ../..

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" Cargo.toml

# build and test
# Note that cargo build must, counterintuitively, come after setup.py install,
# because numpy_weld_convertor.cpp is built by cargo.
make -C weld_rt/cpp/
cargo clean
cargo build --release
cargo test

export LD_LIBRARY_PATH=`pwd`/target/release
python python/grizzly/tests/grizzly_test.py
python python/grizzly/tests/numpy_weld_test.py
deactivate
