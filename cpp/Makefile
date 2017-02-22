OS = $(shell uname -s)
LLVM_VERSION = 3.6

CFLAGS = -O3 -Wall -fno-use-cxa-atexit -fPIC
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= /usr/local/bin/clang-${LLVM_VERSION}
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= /usr/bin/clang-${LLVM_VERSION}
else
  $(error Unsupported platform: ${OS})
endif


all: libpar.a 

parlib.o: parlib.cpp
	${CLANG} ${CFLAGS} -c $< -o $@ 

vb.o: vb.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

libpar.a: parlib.o vb.o
	ar rcs $@ $^

clean:
	rm -f *.bc *.a *.o

.phony: all

