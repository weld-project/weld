# Grizzly

Grizzly is a subset of the Pandas data analytics library integrated with Weld. This document walks through a simple example of how to use Grizzly in an application.

## Prerequisites

This README file assumes for convenience that `WELD_HOME` is set to the root `weld/` root directory.

```bash
$ export WELD_HOME=/path/to/weld/root/directory
```

Build and run tests for Weld (the instructions for this are in `$WELD_HOME/README.md`).  Make sure that `weld` and `grizzly` are correctly setup as detailed in `$WELD_HOME/python/README.md`.


## Running Grizzly's Unit Tests

To run unit tests, run the following:

```bash
$ python $WELD_HOME/python/grizzly/tests/grizzly_test.py     # For Grizzly tests
$ python $WELD_HOME/python/grizzly/tests/numpy_weld_test.py  # For NumPy tests
```

## Using Grizzly in an application

### Data acquisition

To get data for this tutorial run:

```bash
$ wget https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv
```

### A Step-by-Step Walkthrough

First, import the Pandas library and Grizzly:

```bash
$ python
>>> import pandas as pd
>>> import grizzly.grizzly as gr
```

Grizzly depends on native Pandas for file I/O, so to read from a file, call Pandas' `read_csv` function. For the purposes of this tutorial, let's read from a CSV file called `311-service-requests.csv`:

```bash
>>> na_values = ['NO CLUE', 'N/A', '0']
>>> raw_requests = pd.read_csv('311-service-requests.csv', na_values=na_values, dtype={'Incident Zip': str})
```

Grizzly exposes a `DataFrameWeld` object that serves as a wrapper around the native Pandas `DataFrame` object; all of `DataFrameWeld`'s exposed methods are lazily-evaluated (that is, execution is only forced when the `evaluate()` method is called). To create a `DataFrameWeld` object from the `DataFrame` we just read:

```bash
>>> requests = gr.DataFrameWeld(raw_requests)
```

We can then use standard Pandas operators on this `DataFrameWeld` object. `requests` has a column of zipcodes; some of these are "00000". To convert them all to `nan`, we can first compute a predicate using the `==` operator (which returns a `SeriesWeld` object that wraps a native Pandas `Series` object), and then subsequently mask:

```bash
>>> zero_zips = requests['Incident Zip'] == '00000'
>>> requests['Incident Zip'][zero_zips] = "nan"
```

To see all resulting unique zipcodes, we could do:

```bash
>>> result = requests['Incident Zip'].unique()
```

Note that `unique` returns a `LazyOp` object. To convert to a standard NumPy array (that is, to force execution), call:

```bash
>>> print result.evaluate()
```

More examples of workloads that make use of Grizzly are in the [examples/python/grizzly](https://github.com/weld-project/weld/tree/master/examples/python/grizzly) directory.
