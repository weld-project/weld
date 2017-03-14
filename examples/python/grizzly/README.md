This README file assumes for convenience that `WELD_HOME` is set to the `weld` root directory.


Acquire Data for Demo Workloads
===============================

To get data for `dataCleaning` and other related scripts, run:

```bash
>> cd $WELD_HOME/examples/python/grizzly
>> mkdir -p data
>> wget https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv
>> mv 311-service-requests.csv data/311-service-requests-raw.csv
>> scripts/prune-csv -i data/311-service-requests-raw.csv -l "Incident Zip"
>> scripts/replicate-csv -i data/311-service-requests-raw-pruned.csv -o data/311-service-requests.csv -r 30
```

To get data for `getPopulationStats` and other related scripts, run:

```bash
>> cd $WELD_HOME/examples/python/grizzly
>> mkdir -p data
>> wget https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv
>> mv us_cities_states_counties.csv data/us_cities_states_counties_raw.csv
>> scripts/transform-population-csv -i data/us_cities_states_counties_raw.csv -o data/us_cities_states_counties.csv -r 30
```

Running the Demo Workloads
==========================

The demo workloads are in `$WELD_HOME/examples/python/grizzly`.

Each workload has a corresponding `Grizzly` version. For example, the native Pandas/NumPy data cleaning workload is `dataCleaning`, while the corresponding Grizzly workload is `dataCleaningGrizzly`.


As an example, to compare performance between the native Pandas data cleaning workload and the Weld-ified Pandas data cleaning workload, run:

```bash
>> ./dataCleaning         # Native
>> ./dataCleaningGrizzly  # Grizzly
```

These scripts print out timing information.
