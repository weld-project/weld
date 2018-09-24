#!/bin/bash

LLVM_VERSION=$1
LLVM_SYS_VERSION=$2
PYTHON_VERSION=$3
VENV_HOME=`pwd`/.virtualenv

# create llvm-config symlink
sudo rm -f /usr/bin/llvm-config
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
export WELD_HOME=`pwd`

# source $VENV_HOME/python$PYTHON_VERSION/bin/activate
cd python/pyweld
# python setup.py install
cd ../..

cd python/grizzly
# python setup.py install
cd ../..

# set llvm-sys crate version
sed -i "s/llvm-sys = \".*\"/llvm-sys = \"$LLVM_SYS_VERSION\"/g" Cargo.toml

# build and test
# Note that cargo build must, counterintuitively, come after setup.py install,
# because numpy_weld_convertor.cpp is built by cargo.
make -C weld_rt/cpp/st
cargo build #--release
cargo test

# XXX Disabling the Python tests for now, since they rely on some features this
# backend does not support (e.g., vector comparisons and sorting). We'll
# re-enable these once those features are plugged back in.

# export LD_LIBRARY_PATH=`pwd`/target/release
# python python/grizzly/tests/grizzly_test.py
# python python/grizzly/tests/numpy_weld_test.py

# run tests for nditer - first need to install weldnumpy
# cd python/numpy
# python setup.py install
# python ../../examples/python/nditer/nditer_test.py
# cd ../..

# cd $WELD_HOME/weld-benchmarks; python run_benchmarks.py -b tpch_q1 tpch_q6 vector_sum map_reduce data_cleaning crime_index crime_index_simplified -n 5 -f results.tsv -v -d -p performance.png
# mkdir -p $WELD_HOME/results
# mv performance.png $WELD_HOME/results
# mv results.tsv $WELD_HOME/results
cd $WELD_HOME
# deactivate
