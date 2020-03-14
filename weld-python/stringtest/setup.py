import sysconfig
import numpy
import os

from distutils.core import setup, Extension

strings = Extension('stringtest.strings',
                    include_dirs = ['/usr/local/lib', numpy.get_include(),
                        sysconfig.get_paths()['include']],
                    sources = ['stringtest/strings.cpp'],
                    extra_compile_args = ["-O3", "-fPIC", "-std=c++11", "-march=native", "-g"],
                    language = "c++")

setup (name = 'stringtest',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [strings])
