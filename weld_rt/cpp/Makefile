OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS_RT = -O3 -Wall -fno-use-cxa-atexit -fPIC
CFLAGS_BUILDER = ${CFLAGS_RT} -flto
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= clang-${LLVM_VERSION}
else
  $(error Unsupported platform: ${OS})
endif


all: libparrt.a libparbuilder.bc

parlib.o: parlib.cpp
	${CLANG} ${CFLAGS_RT} -c $< -o $@

vb.o: vb.cpp
	${CLANG} ${CFLAGS_BUILDER} -c $< -o $@

merger.o: merger.cpp
	${CLANG} ${CFLAGS_BUILDER} -c $< -o $@

libparrt.a: parlib.o
	ar rcs $@ $^

libparbuilder.bc: vb.o merger.o
	llvm-link-${LLVM_VERSION} $^ -o $@

clean:
	rm -f *.bc *.a *.o

.phony: all

