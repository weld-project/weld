OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS = -O3 -fno-use-cxa-atexit
CFLAGS += $(shell llvm-config --cppflags) $(shell llvm-config --cxxflags)
LIBDIR = $(shell llvm-config --libdir)
LLVMLIBS = $(shell llvm-config --system-libs) $(shell llvm-config --libs)
LIB = libllvmext.a

ifeq ($(OS), Darwin)
  CLANG ?= clang++
else ifeq ($(OS), Linux)
  CLANG ?= clang++-$(LLVM_VERSION)
else
  $(error Unsupported platform: $(OS))
endif

.PHONY: all test

all: $(LIB)

test: $(LIB) test.cpp
	$(CLANG) $(CFLAGS) -L$(LIBDIR) $(LLVMLIBS) $(LIB) test.cpp -o test

llvmext.o: llvmext.cpp
	$(CLANG) $(CFLAGS) -c $< -o $@

$(LIB): llvmext.o
	ar rcs $@ $^

clean:
	rm -f *.o $(LIB)
