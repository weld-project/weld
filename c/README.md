# Weld C API

This directory contains the C headers for the Weld API.

To use in a C program:

```C
#include "weld.h"
```

and when building:

```bash
$ clang -lweld my_program.c
```

Make sure `libweld` is on the `LD_LIBRARY_PATH`:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libweld
```

See the [API documentation](https://github.com/weld-project/weld/blob/master/docs/api.md) for details on the API.
