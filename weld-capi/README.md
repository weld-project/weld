
## Weld C API

This crate compiles a shared library and a static library that can be linked with C/C++ programs. It generates a `weld.h` header file that should be included by the C program:

```c
#include "weld.h"
```

To compile:

```shell
gcc -lweld myprogram.c
```
