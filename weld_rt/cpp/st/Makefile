OS = $(shell uname -s)
LLVM_VERSION = $(shell llvm-config --version | cut -d . -f 1,2)

CFLAGS = -O3 -std=c++11 -Wall -fno-use-cxa-atexit -fPIC
LIB = libdictst.a

ifeq ($(OS), Darwin)
  # macOS - clang++ is sym-linked to clang-6.0
  CLANG ?= clang-$(LLVM_VERSION)
else ifeq ($(OS), Linux)
  # Linux
  CLANG ?= clang++-$(LLVM_VERSION)
else
  $(error Unsupported platform: $(OS))
endif

all: $(LIB)

debug: CFLAGS += -DDEBUG -g
debug: all

dict-st.o: dict-st.cpp
	$(CLANG) $(CFLAGS) -c $< -o $@

$(LIB): dict-st.o
	ar rcs $@ $^

clean:
	rm -f *.bc *.o *.a

.phony: all debug
