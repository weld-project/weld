# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

### Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`).

### Setup

To setup Weld's Python API, add the following to the `PYTHONPATH`:
```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/python  # by default, $WELD_HOME/python
```

Also, make sure that `libweld` and `libweldrt` are on  the `LD_LIBRARY_PATH`.
