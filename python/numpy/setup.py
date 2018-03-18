import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext
from setuptools.command.install import install

class Install(install):
    def run(self):
        install.run(self)
        python_executable = sys.executable

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='weldnumpy',
      version='0.0.1',
      packages=['weldnumpy'],
      cmdclass={"install": Install},
      distclass=BinaryDistribution,
      url='https://github.com/parimarjan/weld/tree/master/python/numpy',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pyweld', 'pygrizzly', 'numpy>=1.13', 'scipy', 'pytest'])
