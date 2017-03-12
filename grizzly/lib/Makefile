OS=$(shell uname -s)
LLVM_VERSION=3.6

ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang
  CLANGPP ?= clang++
  FLAGS=-march=native -O3 -flto
  DFLAGS=-dynamiclib
  DYLIB_SUFFIX=.dylib
  PYTHON_INCLUDES=-I/usr/include/python2.7 -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= /usr/bin/clang-${LLVM_VERSION}
  CLANGPP ?= /usr/bin/clang++-${LLVM_VERSION}
  FLAGS=-mavx2 -fuse-ld=gold -flto
  DFLAGS=-shared -fPIC
  DYLIB_SUFFIX=.so
  PYTHON_INCLUDES=-I/usr/include/python2.7 -I/home/rpalamut/anaconda2/python2.7 -I/home/rpalamut/anaconda2/lib/python2.7/site-packages/numpy/core/include
else
  $(error Unsupported platform: ${OS})
endif

convertor:
	${CLANGPP} ${DFLAGS} -w -march=native -lpython2.7 numpy_weld_convertor.cpp -o numpy_weld_convertor${DYLIB_SUFFIX}  ${PYTHON_INCLUDES}

clean:
	rm -rf run numpy_weld_convertor${DYLIB_SUFFIX} *.pyc
