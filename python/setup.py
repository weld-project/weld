from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext as _build_ext
from os.path import dirname, exists, join as pjoin
from os import remove
import numpy
import pyarrow

class build_ext(_build_ext):

    def run(self):
        _build_ext.run(self)
        cpp_path = pjoin('weld', 'arrow_compat.cpp')
        if exists(cpp_path):
            remove(cpp_path)

setup(name='grizzly',
      version='0.0.1',
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      packages=['grizzly', 'weld'],
      ext_modules=cythonize([
        Extension('weld.arrow_compat',
              sources=['weld/arrow_compat.pyx'],
              extra_compile_args=['-std=c++11',
                '-I' + pjoin(dirname(pyarrow.__file__), 'include'),
                '-I' + numpy.get_include()],

              extra_link_args=['-std=c++11', '-stdlib=libc++'],
              include_dirs=['.'],
              language='c++')
        ]),
      cmdclass={
            'build_ext': build_ext
    },
    install_requires=['pandas', 'numpy', 'pyarrow', 'cython'])
