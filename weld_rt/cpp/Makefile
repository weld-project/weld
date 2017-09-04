OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS = -O3 -std=c++11 -Wall -fno-use-cxa-atexit -fPIC
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang++-${LLVM_VERSION}
	DYLIB = libweldrt.dylib
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= clang++-${LLVM_VERSION}
	DYLIB = libweldrt.so
else
  $(error Unsupported platform: ${OS})
endif

all: ${DYLIB} inline.bc

inline.o: inline.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

inline.bc: inline.cpp
	${CLANG} ${CFLAGS} -c -emit-llvm $< -o $@

vb.o: vb.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

merger.o: merger.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

runtime.o: runtime.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

${DYLIB}: vb.o merger.o runtime.o inline.o
	${CLANG} -shared -o $@ $^
	mkdir -p ../../target/release
	cp ${DYLIB} ../../target/release
	mkdir -p ../../target/debug
	cp ${DYLIB} ../../target/debug

clean:
	rm -f *.bc *.o ${DYLIB}

.phony: all

