#!/bin/bash

grep -R "unimplemented" weld/codegen/llvm2 | grep -v "Binary"
