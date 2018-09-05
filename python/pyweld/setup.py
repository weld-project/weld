import os
import platform
import shutil
import subprocess

import setuptools.command.build_ext as _build_ext
from setuptools import setup, Distribution
from setuptools.command.develop import develop
from setuptools.extension import Extension

system = platform.system()
if system == 'Linux':
    libweld = "libweld.so"
elif system == 'Windows':
    libweld = "libweld.dll"
elif system == 'Darwin':
    libweld = "libweld.dylib"
else:
    raise OSError("Unsupported platform {}", system)

libweld_dir = os.environ["WELD_HOME"] + "/target/release/"
libweld = libweld_dir + libweld
module1 = Extension('weld', sources=[], libraries=['weld'], library_dirs=[libweld_dir])

is_develop_command = False
class build_ext(_build_ext.build_ext):
    def run(self):
        if not os.path.exists(libweld):
            subprocess.call("cargo build --release", shell=True)
        if not is_develop_command:
            self.move_file(libweld, "weld")

    def move_file(self, filename, directory):
        source = filename
        dir, name = os.path.split(source)        
        destination = os.path.join(self.build_lib + "/" + directory + "/", name)
        print("Copying {} to {}".format(source, destination))
        shutil.copy(source, destination)

class Develop(develop):
    def run(self):
        global is_develop_command
        if not os.path.exists(libweld):
            subprocess.call("cargo build --release", shell=True)
        self.move_file(libweld, "weld")
        is_develop_command = True
        develop.run(self)

    def move_file(self, filename, directory):
        source = filename
        dir, name = os.path.split(source)
        destination = os.path.join(directory + "/", name)
        print("Copying {} to {}".format(source, destination))
        shutil.copy(source, destination)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='pyweld',
      version='0.0.6',
      packages=['weld'],
      cmdclass={"build_ext": build_ext, "develop": Develop},
      distclass=BinaryDistribution,
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pandas', 'numpy'],
      ext_modules=[module1])
