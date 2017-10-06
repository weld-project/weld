CC=clang++
FLAGS=-O0 -g -Iinclude

.PHONY: all

all:
	$(CC) $(FLAGS) harness.cpp -L../../target/release/ -I.. -I../../c/ -lweld -o bench

clean:
	rm -rf bench bench.dSYM
