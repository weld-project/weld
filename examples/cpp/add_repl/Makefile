CC=clang++
LDFLAGS=-L../../../target/debug/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} add_repl.cpp -o run

clean:
	rm -rf run

