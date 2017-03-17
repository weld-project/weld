# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

### Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`).
Make sure the `WELD_HOME` environment variable is set as detailed in the main
`README.md`.

### Setup

If you want to install Weld's Python API and Grizzly in 'development' mode, run:
```bash
$ python setup.py develop
```

Otherwise, run:
```bash
$ python setup.py install
```

Also, make sure that `libweld` and `libweldrt` are on  the `LD_LIBRARY_PATH`.

### Updating Weld and Grizzly

If you installed Weld's Python API and Grizzly in 'development' mode, run:
```bash
$ git pull
```

If you installed Weld's Python API and Grizzly in 'install' mode, run:
```bash
$ git pull
$ python setup.py install
```
