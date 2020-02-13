#!/bin/bash

set -e

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH=$PATH:$HOME/.cargo/bin

# For Python tests.
rustup toolchain install nightly

rustup component add clippy
rustup component add rustfmt
