# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

## Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`).

## Setup

To setup Weld's Python API, add the following to the `PYTHONPATH`:
```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/python  # by default, $WELD_HOME/python
```
## Running Grizzly's Unit Tests

To run unit tests, run the following:

```bash
$ python $WELD_HOME/python/grizzly/tests/grizzlyTest    # For Grizzly tests
$ python $WELD_HOME/python/grizzly/tests/numpyWeldTest  # For NumPy tests
```
