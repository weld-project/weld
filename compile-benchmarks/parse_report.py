#!/usr/bin/env python

#
# Prints out a Weld log file with a compilation report in CSV format.
#
# Usage:
# parse_report.py <filename>

import sys
import os

if len(sys.argv) != 2:
    print "Usage: {} <filename>".format(sys.argv[0])
    sys.exit(1)

# Read in the log file
with open(sys.argv[1]) as f:
    lines = [line.strip() for line in f]

# Find where the compilation report begins
start = 0
while lines[start].find("Weld Compiler") == -1:
    start += 1

lines = lines[start:]

times = []

for line in lines:
    tokens = line.strip().rsplit(" ", 1)
    if tokens[-1] == "ms":
        assert len(tokens) == 2
        cleaned = [token.strip() for token in tokens[0].split(":")]
        if len(cleaned) == 2:
            times.append(cleaned)

keys, times = zip(*times)

print ",".join(keys)
print ",".join(times)
