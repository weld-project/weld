CC=clang++
LDFLAGS=-L../../../target/debug/
LIBS=-lcweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} composite_types.cpp -o run

clean:
	rm -rf run

