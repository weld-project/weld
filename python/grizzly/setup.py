import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext

system = platform.system()
if system == 'Linux':
    numpy_convertor = "numpy_weld_convertor.so"
elif system == 'Windows':
    numpy_convertor = "numpy_weld_convertor.dll"
elif system == 'Darwin':
    numpy_convertor = "numpy_weld_convertor.dylib"
else:
    raise OSError("Unsupported platform {}", system)

class build_ext(_build_ext.build_ext):
    def run(self):
        if not os.path.exists("grizzly/" + numpy_convertor):
            subprocess.call("cd grizzly && make && cd ..", shell=True)
        self.move_file("grizzly/" + numpy_convertor, "grizzly")

    def move_file(self, filename, directory):
        source = filename
        dir, name = os.path.split(source)        
        destination = os.path.join(self.build_lib + "/" + directory + "/", name)
        print("Copying {} to {}".format(source, destination))
        shutil.copy(source, destination)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='grizzly',
      version='0.0.1',
      packages=['grizzly'],
      cmdclass={"build_ext": build_ext},
      distclass=BinaryDistribution,
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pyweld'])
