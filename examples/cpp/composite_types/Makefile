CC=clang++
LDFLAGS=-L../../../target/release/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} composite_types.cpp -o run

clean:
	rm -rf run

