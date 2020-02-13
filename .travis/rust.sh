#!/bin/bash

set -e

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
export PATH=$PATH:$HOME/.cargo/bin

# For Python tests.
rustup toolchain install nightly

rustup component add clippy
rustup component add rustfmt
