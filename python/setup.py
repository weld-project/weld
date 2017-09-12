import os
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext

libweld = "../target/release/libweld.dylib"

class build_ext(_build_ext.build_ext):
    def run(self):
        if not os.path.exists(libweld):
            subprocess.call("cargo build --release", shell=True)
        self.move_file(libweld)

    def move_file(self, filename):
        source = filename
        dir, name = os.path.split(source)        
        destination = os.path.join(self.build_lib + "/weld/", name)
        print("Copying {} to {}".format(source, destination))
        shutil.copy(source, destination)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='weld',
      version='0.0.1',
      packages=['weld'],
      cmdclass={"build_ext": build_ext},
      distclass=BinaryDistribution,
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pandas', 'numpy'])
