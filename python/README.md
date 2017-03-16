# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

### Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`).

### Setup

To setup Weld's Python API, add the following to the `PYTHONPATH`:
```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/python  # by default, $WELD_HOME/python
```

## Running Grizzly

### Acquire Data

To get data for this tutorial run:

```bash
$ mkdir -p data
$ wget https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv
$ $WELD_HOME/examples/python/grizzly/scripts/prune-csv -i 311-service-requests.csv -l "Incident Zip"
$ $WELD_HOME/examples/python/grizzly/scripts/replicate-csv -i 311-service-requests-pruned.csv -o 311-service-requests.csv -r 30
```

### Using Grizzly in a Python REPL

Import the pandas library and grizzly.

```bash
$ python
> import pandas as pd
> import grizzly.grizzly as gr
> import time
```

We first use pandas for reading CSV.

Then we use the DataFrameWeld wrapper to use grizzly. 

```bash
> na_values = ['NO CLUE', 'N/A', '0']
> raw_reqess = pd.read_csv('311-service-requests.csv', na_values=na_values, dtype={'Incident Zip': str})
> requests = gr.DataFrameWeld(raw_requests)
```

Next we use common pandas expressions for data cleaning.

```bash
> zero_zips = requests['Incident Zip'] == '00000'
> requests['Incident Zip'][zero_zips] = "nan"
> result = requests['Incident Zip'].unique()
```

Note that unlike pandas, grizzly performs these operations lazily.

We need to call evaluate() to materialize the result.

```bash
> print result.evaluate()
```

