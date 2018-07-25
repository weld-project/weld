#!/bin/bash

grep -R "unimplemented" . | grep -v "Binary"
