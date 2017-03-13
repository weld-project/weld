Prerequisites
=============

Make sure you're able to build and run tests for Weld (instructions are at `../README.md`.
Don't forget to set the `WELD_HOME` environment variable


Build instructions
==================

To get Grizzly to work, set the following environment variables,
```bash
>> export GRIZZLY_HOME=~/path/to/grizzly
>> export GRIZZLY_LIB_HOME=$GRIZZLY_HOME/lib
```


How to run unit tests
=====================

From the parent Weld directory, run
```bash
>> python grizzly/tests/grizzlyTest    # For Grizzly tests
>> python grizzly/tests/numpyWeldTest  # For NumPy tests
```
to run unit tests


How to get data
===============

To get data for `dataCleaning` and other related scripts, run
```bash
>> cd $GRIZZLY_HOME
>> mkdir -p data
>> wget https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv
>> mv 311-service-requests.csv data/311-service-requests-raw.csv
>> scripts/prune-csv -i data/311-service-requests-raw.csv -l "Incident Zip"
>> scripts/replicate-csv -i data/311-service-requests-raw-pruned.csv -o data/311-service-requests.csv -r 30
```

To get data for `getPopulationStats` and other related scripts, run
```bash
>> cd $GRIZZLY_HOME
>> mkdir -p data
>> wget https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv
>> mv us_cities_states_counties.csv data/us_cities_states_counties_raw.csv
>> scripts/transform-population-csv -i data/us_cities_states_counties_raw.csv -o data/us_cities_states_counties.csv -r 30
```


How to run
==========

For every native Pandas / NumPy workload `x`, the equivalent workload that uses
NumPy and Pandas is `xGrizzly`.

For example, to compare performance between the native Pandas data cleaning workload
and the Weld-ified Pandas data cleaning workload, run
```bash
>> cd $GRIZZLY_HOME
>> ./dataCleaning         # Native
>> ./dataCleaningGrizzly  # Grizzly
```
