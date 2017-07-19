#!/bin/bash

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3
VENV_HOME=`pwd`/.virtualenv

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
export WELD_HOME=`pwd`

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" easy_ll/Cargo.toml

cargo clean
cargo build --release
cargo test

unset WELD_HOME
source $VENV_HOME/python$PYTHON_VERSION/bin/activate
cd python
python setup.py install
cd ..

python python/grizzly/tests/grizzly_test.py
python python/grizzly/tests/grizzly_test_arrow.py
python python/grizzly/tests/numpy_weld_test.py
deactivate
