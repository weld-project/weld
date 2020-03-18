#!/usr/bin/env python

import numpy
import sysconfig
import sys

from setuptools import setup, Extension
from setuptools_rust import RustExtension

setup_requires = ["numpy", "setuptools-rust>=0.10.1", "wheel"]
install_requires = ["pandas"]

# Encoding and decoding for strings.
stringencdec = Extension('weld.encoders._strings',
                    include_dirs = ['/usr/local/lib', numpy.get_include(),
                        sysconfig.get_paths()['include']],
                    sources = ['weld/encoders/strings.cpp'],
                    extra_compile_args = ["-O3", "-std=c++11", "-march=native"],
                    language = "c++")

setup(
    name="weld",
    version="0.1.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=["weld"],
    rust_extensions=[RustExtension("weld.core")],
    ext_modules = [stringencdec],
    setup_requires=setup_requires,
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
