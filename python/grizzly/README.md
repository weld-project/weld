# Grizzly

This README file assumes for convenience that `WELD_HOME` is set to the `weld` root directory.


### Prerequisites

Build and run tests for Weld (the instructions for this are in `$WELD_HOME/README.md`).  Make sure the `PYTHONPATH` is set correctly as detailed in `$WELD_HOME/python/README.md`.


### Running Grizzly's Unit Tests

To run unit tests, run the following:

```bash
$ python $WELD_HOME/python/grizzly/tests/grizzlyTest    # For Grizzly tests
$ python $WELD_HOME/python/grizzly/tests/numpyWeldTest  # For NumPy tests
```

## Running Grizzly

### Acquire Data

To get data for this tutorial run:

```bash
$ wget https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv
$ $WELD_HOME/examples/python/grizzly/scripts/prune-csv -i 311-service-requests.csv -l "Incident Zip"
$ $WELD_HOME/examples/python/grizzly/scripts/replicate-csv -i 311-service-requests-pruned.csv -o 311-service-requests.csv -r 30
```

### Using Grizzly in a Python REPL

Import the Pandas library and Grizzly.

```bash
$ python
> import pandas as pd
> import grizzly.grizzly as gr
> import time
```

We first use Pandas for reading CSV.

Then we use the DataFrameWeld wrapper to use Grizzly. 

```bash
> na_values = ['NO CLUE', 'N/A', '0']
> raw_reqess = pd.read_csv('311-service-requests.csv', na_values=na_values, dtype={'Incident Zip': str})
> requests = gr.DataFrameWeld(raw_requests)
```

Next we use common Pandas expressions for data cleaning.

```bash
> zero_zips = requests['Incident Zip'] == '00000'
> requests['Incident Zip'][zero_zips] = "nan"
> result = requests['Incident Zip'].unique()
```

Note that unlike Pandas, Grizzly performs these operations lazily.

We need to call evaluate() to materialize the result.

```bash
> print result.evaluate()
```