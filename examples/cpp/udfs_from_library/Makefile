CC=clang++
LDFLAGS=-L../../../target/debug/
LIBS=-lweld

OS = $(shell uname -s)
ifeq (${OS}, Darwin)
  LIB_FILE=libudf.dylib
else ifeq (${OS}, Linux)
  LIB_FILE=libudf.so
else
  $(error Unsupported platform: ${OS})
endif

.PHONY: all clean

all: run ${LIB_FILE}

run: udfs.cpp
	${CC} -DUDFLIB="\"${LIB_FILE}\"" ${LDFLAGS} ${LIBS} udfs.cpp -o run

${LIB_FILE}: lib.cpp
	${CC} -shared -o ${LIB_FILE} lib.cpp

clean:
	rm -rf run ${LIB_FILE}

