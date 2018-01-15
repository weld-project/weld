OS=$(shell uname -s)
LLVM_VERSION=$(shell llvm-config --version | cut -d . -f 1,2)

PYTHON_HEADER_INCLUDE = $(shell python-config --includes)
NUMPY_HEADER_INCLUDE = -I$(shell python -c "import numpy; print numpy.get_include()" 2>/dev/null)

ifndef EXEC
  EXEC=python
endif

PYTHON_LDFLAGS=-L$(shell $(EXEC) -c "from distutils import sysconfig; print sysconfig.get_config_var('LIBDIR')" 2>/dev/null) $(shell python-config --libs)

ifeq (${NUMPY_HEADER_INCLUDE}, -I)
  $(error Error: NumPy installation not found)
endif

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
	${CLANG} ${DFLAGS} -w -march=native $(PYTHON_LDFLAGS) numpy_weld_convertor.cpp -o numpy_weld_convertor${DYLIB_SUFFIX}  ${PYTHON_INCLUDES}

clean:
	rm -rf run numpy_weld_convertor${DYLIB_SUFFIX} *.pyc
