Build instructions
==================

- Follow the instructions in `../README.md` to build Weld (don't forget
  to set the `WELD_HOME` environment variable)
- In addition, add the following two environment variables
  ```export GRIZZLY_HOME=~/path/to/grizzly
  export GRIZZLY_LIB_HOME=$GRIZZLY_HOME/lib```


How to run unit tests
=====================

From the parent Weld directory, run `python grizzly/tests/grizzlyTest` and
`python grizzly/tests/numpyWeldTest` to run unit tests for our Grizzly and
NumPy ports respectively


How to get data
===============

- Data for dataCleaning and related scripts is
  [here](https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv)
- Data for getPopulationStats and related scripts is
  [here](https://github.com/grammakov/USA-cities-and-states/tree/master)
- Put data files in a `data` directory created in this folder


What to run
===========

- Run `scripts/plot-all-experiments-bar-graph` to produce bar graph summarizing
  different schemes for a bunch of different workloads
- Run `./dataCleaning` for native Pandas implementation of simple data cleaning
  workload
- Run `./dataCleaningLazy` Pandas implementation with lazy evaluation and NVL-
  generated code
- Run `cd c++; ./dataCleaning` for semi-optimized C++ implementation of
  the same workload
- Run `tests/pandasNVLTest` to run unit-test suite
