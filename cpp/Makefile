OS = $(shell uname -s)
LLVM_VERSION = 3.6

CFLAGS = -Wall -fno-use-cxa-atexit
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= /usr/local/bin/clang-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= /usr/bin/clang-${LLVM_VERSION}
  CFLAGS += -fno-use-cxa-get-exception-ptr
else
  $(error Unsupported platform: ${OS})
endif


all: parlib.bc

parlib.bc: parlib.cpp
	${CLANG} ${CFLAGS} -c -emit-llvm parlib.cpp

clean:
	rm -f *.bc

.phony: all

