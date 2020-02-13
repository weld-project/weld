#!/bin/bash

### Install LLVM.

LLVM_VERSION=$1
DIST=$2

sudo -E apt-add-repository -y "ppa:deadsnakes/ppa"
curl -sSL "https://apt.llvm.org/llvm-snapshot.gpg.key"\
  | sudo -E apt-key add -
echo "deb http://apt.llvm.org/$DIST/ llvm-toolchain-$DIST-$LLVM_VERSION main"\
  | sudo tee -a /etc/apt/sources.list > /dev/null
curl -sSL "https://build.travis-ci.org/files/gpg/travis-security.asc"\
  | sudo -E apt-key add -
echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu $DIST main"\
  | sudo tee -a /etc/apt/sources.list > /dev/null

sudo -E apt-get\
  --no-install-suggests\
  --no-install-recommends\
  -qy\
  --allow-unauthenticated\ install\
  gcc make llvm-$LLVM_VERSION llvm-$LLVM_VERSION-dev clang-$LLVM_VERSION lib32z1-dev

which llvm-config
llvm-config --libdir
echo $PATH

# Remove the old `llvm-config`
sudo rm -f `which llvm-config`
# create new llvm-config symlink
sudo ln -s /usr/bin/llvm-config-$LLVM_VERSION /usr/bin/llvm-config
