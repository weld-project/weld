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

Make sure `WELD_HOME` is set to the root Weld directory.

See the [API documentation](https://github.com/weld-project/weld/blob/master/docs/api.md) for details on the API.
