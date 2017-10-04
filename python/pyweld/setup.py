import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup, Distribution
import setuptools.command.build_ext as _build_ext

system = platform.system()
if system == 'Linux':
    libweld = "libweld.so"
elif system == 'Windows':
    libweld = "libweld.dll"
elif system == 'Darwin':
    libweld = "libweld.dylib"
else:
    raise OSError("Unsupported platform {}", system)

libweld = os.environ["WELD_HOME"] + "/target/release/" + libweld


class build_ext(_build_ext.build_ext):
    def run(self):
        if not os.path.exists(libweld):
            subprocess.call("cargo build --release", shell=True)
        self.move_file(libweld, "weld")

    def move_file(self, filename, directory):
        source = filename
        dir, name = os.path.split(source)        
        destination = os.path.join(self.build_lib + "/" + directory + "/", name)
        print("Copying {} to {}".format(source, destination))
        shutil.copy(source, destination)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(name='pyweld',
      version='0.0.1',
      packages=['weld'],
      cmdclass={"build_ext": build_ext},
      distclass=BinaryDistribution,
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      install_requires=['pandas', 'numpy'])
