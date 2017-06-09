#! /bin/bash

# Returns a non-zero exit code if `rustfmt` produces a diff.

if [[ $(cargo fmt -- --write-mode=diff 2> /dev/null) ]]; then
    exit 1
else
    exit 0
fi
