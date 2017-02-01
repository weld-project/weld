CC=clang++
LDFLAGS=-L../../../../target/debug/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} mem_mgmt.cpp -o run

clean:
	rm -rf run

