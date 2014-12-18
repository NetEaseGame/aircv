# coding: utf-8
from __future__ import print_function

import aircv

try: from distutils.core import setup
except ImportError: from setuptools import setup

setup(
      name='pyignore',
      version=aircv.__version__,
      license="MIT",
      description='Image utils based on python-opencv2',

      author='codeskyblue',
      author_email='codeskyblue@gmail.com',
      url='http://github.com/netease/aircv',

      py_modules=['aircv'],
      install_requires=[],
)
