OS=$(shell uname -s)
LLVM_VERSION=3.8

PYTHON_HEADER_INCLUDE = $(shell python-config --includes)
NUMPY_HEADER_INCLUDE = -I$(shell python -c "import numpy; print numpy.get_include()")
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang
  CLANGPP ?= clang++
  FLAGS=-march=native -O3 -flto
  DFLAGS=-dynamiclib
  DYLIB_SUFFIX=.dylib
  PYTHON_INCLUDES=${PYTHON_HEADER_INCLUDE} ${NUMPY_HEADER_INCLUDE}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= /usr/bin/clang-${LLVM_VERSION}
  CLANGPP ?= /usr/bin/clang++-${LLVM_VERSION}
  FLAGS=-mavx2 -fuse-ld=gold -flto
  DFLAGS=-shared -fPIC
  DYLIB_SUFFIX=.so
  PYTHON_INCLUDES=${PYTHON_HEADER_INCLUDE} ${NUMPY_HEADER_INCLUDE}
else
  $(error Unsupported platform: ${OS})
endif

convertor:
	${CLANGPP} ${DFLAGS} -w -march=native -lpython2.7 numpy_weld_convertor.cpp -o numpy_weld_convertor${DYLIB_SUFFIX}  ${PYTHON_INCLUDES}

clean:
	rm -rf run numpy_weld_convertor${DYLIB_SUFFIX} *.pyc
