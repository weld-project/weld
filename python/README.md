# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

### Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`).
Make sure the `WELD_HOME` environment variable is set as detailed in the main
`README.md`.

### Setup

If you want to install Weld's Python API and Grizzly in 'development' mode, run:
```bash
$ cd pyweld;  python setup.py develop; cd ..
$ cd grizzly; python setup.py develop; cd ..
```

Otherwise, run:
```bash
$ cd pyweld;  python setup.py install; cd ..
$ cd grizzly; python setup.py install; cd ..
```

Alternatively, you can install our pre-build PyPI packages,
```bash
$ pip install pyweld
$ pip install pygrizzly
```

### Updating Weld and Grizzly

If you installed Weld's Python API and Grizzly in 'development' mode, run:
```bash
$ git pull
```

If you installed Weld's Python API and Grizzly in 'install' mode, run:
```bash
$ git pull
$ cd pyweld;  python setup.py install; cd ..
$ cd grizzly; python setup.py install; cd ..
```
