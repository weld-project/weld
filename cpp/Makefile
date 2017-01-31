OS = $(shell uname -s)
LLVM_VERSION = 3.6

ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= /usr/local/bin/clang-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= /usr/bin/clang-${LLVM_VERSION}
else
  $(error Unsupported platform: ${OS})
endif

CFLAGS = -Wall -fno-use-cxa-atexit

all: parlib.bc

parlib.bc: parlib.cpp
	${CLANG} ${CFLAGS} -DW=${W} -c -emit-llvm parlib.cpp

clean:
	rm -f *.bc

.phony: all

