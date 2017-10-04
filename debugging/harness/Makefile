CC=clang++
FLAGS=-O0 -g -Iinclude -I../weld_rt/cpp 

.PHONY: all

all:
	$(CC) $(FLAGS) -I../weld_rt/cpp -I.. harness.cpp crashing2.ll ../../weld_rt/cpp/runtime.cpp ../../weld_rt/cpp/vb.cpp ../../weld_rt/cpp/merger.cpp ../../weld_rt/cpp/inline.cpp -o bench

clean:
	rm -rf bench bench.dSYM
