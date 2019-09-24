from setuptools import setup
from setuptools_rust import Binding, RustExtension

"""
setup(name='weld',
      version='0.3.1',
      packages=['weld'],
      url='https://github.com/weld-project/weld',
      author='Weld Developers',
      author_email='weld-group@lists.stanford.edu',
      ext_modules=[module1])
"""

setup(name='weld',
      version="0.3.1",
      rust_extensions=[RustExtension('weld', '../../weld-capi/Cargo.toml',  binding=Binding.NoBinding)],
      zip_safe=False)
