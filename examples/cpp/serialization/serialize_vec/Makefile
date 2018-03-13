CC=clang++
LDFLAGS=-L../../../../target/debug/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} -O3 ${LDFLAGS} ${LIBS} serialize_test.cpp -o run

clean:
	rm -rf run
