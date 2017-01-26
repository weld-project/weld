Build instructions
==================

- Download Boost from [SourceForge](http://www.boost.org/users/history/version_1_61_0.html)
- In the directory you want to put the Boost installation, execute
`tar --bzip2 -xf /path/to/boost_1_61_0.tar.bz2`
- Now, add the following environment variable definition,
`BOOST_HOME=/directory/where/boost/was/extracted`
- Now, `cd c++; make`
- In addition, add the following two environment variable definitions
  to run the various benchmark scripts,
  ```export PANDAS_TEST_HOME=~/path/to/nvl/root/llvmrunner/tests/pandastest
  export PANDAS_NVL_HOME=$PANDAS_TEST_HOME/lib```


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
