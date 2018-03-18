import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop

system = platform.system()
if system == 'Linux':
    numpy_convertor = "numpy_weld_convertor.so"
elif system == 'Windows':
    numpy_convertor = "numpy_weld_convertor.dll"
elif system == 'Darwin':
    numpy_convertor = "numpy_weld_convertor.dylib"
else:
    raise OSError("Unsupported platform {}", system)

class Install(install):
    def run(self):
        install.run(self)
        python_executable = sys.executable
        protoc_command_clean = ["make -C " + self.install_lib + "grizzly clean"]
        if subprocess.call(protoc_command_clean, shell=True) != 0:
            sys.exit(-1)

        protoc_command_make = ["make -C " + self.install_lib + "grizzly/ EXEC=" + python_executable]
        if subprocess.call(protoc_command_make, shell=True) != 0:
            sys.exit(-1)

class Develop(develop):
    def run(self):
        develop.run(self)
        python_executable = sys.executable
        protoc_command_clean = ["make -C grizzly/ clean"]
        if subprocess.call(protoc_command_clean, shell=True) != 0:
            sys.exit(-1)

        protoc_command_make = ["make -C grizzly/ EXEC=" + python_executable]
        if subprocess.call(protoc_command_make, shell=True) != 0:
            sys.exit(-1)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='pygrizzly',
      version='0.0.4',
      packages=['grizzly'],
      package_data = {'grizzly': ['numpy_weld_convertor.cpp', 'common.h', 'Makefile']},
      cmdclass={"install": Install, "develop": Develop},
      distclass=BinaryDistribution,
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pyweld'])
