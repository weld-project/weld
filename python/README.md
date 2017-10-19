# Weld Python API

This directory contains the Weld Python API and the Grizzly implementation.

### Prerequisites

Build and run tests for Weld (the instructions for this are in the main `README.md`. <br />
Make sure the `WELD_HOME` environment variable is set as detailed in the main
`README.md`.

### Setup

#### Installing Weld's Python API

If you want to install Weld's Python API in `development` mode, run:
```bash
$ cd pyweld;  python setup.py develop; cd ..
```

Otherwise, run:
```bash
$ cd pyweld;  python setup.py install; cd ..
```

Note that setup.py will call `cargo build --release`. <br />
The generated libweld binary will be copied to the appropriate directory. <br />
In developer mode, `libweld` will be copied to the weld directory. <br /> 
In install mode, `libweld` will be copied to to build/lib.macosx-10.7-x86_64-2.7/weld. <br />
Note that the tag name following `lib` depends on both the platform and python version. <br />

Alternatively, you can install our pre-build PyPI packages,

```bash
$ pip install pyweld
$ pip install pygrizzly
```

#### Installing Grizzly

If you want to install Grizzly in 'development' mode, run:
```bash
$ cd grizzly; python setup.py develop; cd ..
```

Otherwise, run:
```bash
$ cd grizzly; python setup.py install; cd ..
```

Grizzly is packaged as a source dstribution and so will include `numpy_weld_convertor.cpp`, `common.h`, and `Makefile`.
During installation setup.py will call `make` to compile the numpy_weld_convertor dynamic library.

Note: You must have Weld's Python API installed before installing Grizzly.

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

### Troubleshooting

The python bindings and grizzly have been tested on OSX, and Ubuntu 16.04 and 14.04.
You may run into the following runtime error:

```bash
 /System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site.py", line 231, in getuserbase
   USER_BASE = get_config_var('userbase')
 File "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/sysconfig.py", line 520, in get_config_var
   return get_config_vars().get(name)
 File "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/sysconfig.py", line 453, in get_config_vars
   import re
 File "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/re.py", line 108, in <module>
   import _locale
SystemError: dynamic module not initialized properly
```

This can be resolved by upgrading pip to the latest version.
If you are using a virtual environment, make sure to also have pip installed within the virtual environment.
The root cause is that the python interpreter used for installation is different from the interpreter used at runtime.
