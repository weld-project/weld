OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS = -O3 -std=c++11 -stdlib=libstdc++ -Wall -fno-use-cxa-atexit -fPIC -flto
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang++-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= clang++-${LLVM_VERSION}
else
  $(error Unsupported platform: ${OS})
endif


all: libweldrt.bc

vb.o: vb.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

merger.o: merger.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

runtime.o: runtime.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

libweldrt.bc: vb.o merger.o runtime.o
	llvm-link-${LLVM_VERSION} $^ -o $@

clean:
	rm -f *.bc *.o

.phony: all

