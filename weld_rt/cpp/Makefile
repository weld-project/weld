OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS = -O3 -std=c++11 -Wall -fno-use-cxa-atexit -fPIC
ifeq (${OS}, Darwin)
  # OS X
  CLANG ?= clang++-${LLVM_VERSION}
	LIB = libweldrt.a
else ifeq (${OS}, Linux)
  # Linux
  CLANG ?= clang++-${LLVM_VERSION}
	LIB = libweldrt.a
else
  $(error Unsupported platform: ${OS})
endif

all: ${LIB}

vb.o: vb.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

merger.o: merger.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

runtime.o: runtime.cpp
	${CLANG} ${CFLAGS} -c $< -o $@

# We combine the different object files into one object file in order to prevent the linker from
# eliminating seemingly unused object files when we create executable like the Repl or the
# integration test executable.
weldrt.o: vb.o merger.o runtime.o
	ld -r $^ -o $@

${LIB}: weldrt.o
	ar rcs $@ $^

clean:
	rm -f *.o *.a

.phony: all
