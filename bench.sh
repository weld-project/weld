#!/bin/bash

# Runs all the benchmarks under the benches target.

if [ $# -eq 0 ]
then
    cargo bench --bench benches
else
    cargo bench --bench benches -- $@
fi

