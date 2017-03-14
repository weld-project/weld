This README file assumes for convenience that `$WELD_HOME` is set to the `weld` root directory.


Prerequisites
=============

Build and run tests for Weld (the instructions for this are in  `$WELD_HOME/README.md`).


Setup
=====

To setup Weld's Python API, add `$WELD_HOME/python` to the `$PYTHONPATH`,like:
```bash
>> export PYTHONPATH=$PYTHONPATH:/path/to/python  # by default, $WELD_HOME/python
```

Running Grizzly's Unit Tests
============================

To run unit tests, run the following:

```bash
>> python $WELD_HOME/python/grizzly/tests/grizzlyTest    # For Grizzly tests
>> python $WELD_HOME/python/grizzly/tests/numpyWeldTest  # For NumPy tests
```
