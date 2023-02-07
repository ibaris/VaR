# -*- coding: utf-8 -*-
"""
Setup of VaR Package
====================
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
from setuptools import find_packages
from setuptools import setup

try:
    import pypandoc

    long_description = pypandoc.convert('README_PYPI.md', 'rst')
except(IOError, ImportError):
    long_description = open('README_PYPI.md').read()


def get_packages():
    find_packages(exclude=['docs']),
    return find_packages()


# ------------------------------------------------------------------------------------------------------------
# Environmental Functions
# ------------------------------------------------------------------------------------------------------------
def get_version():
    """
    Function to get the version.

    Returns
    -------
    out : str
    """
    version = dict()
    with open("var/__version__.py") as fp:
        exec(fp.read(), version)

    return version['__version__'].split("-")[0]


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='var',

      version=get_version(),
      description='Different Methods to Estimate the Value-at-Risk of a portfolio.',
      packages=get_packages(),
      install_requires=required,
      author="Ismail Baris",
      maintainer='Ismail Baris',
      author_email='i.baris@outlook.de',
      url='https://github.com/ibaris/VaR',
      long_description=long_description,

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: End Users/Desktop',
          'Programming Language :: Python :: 3.7',
          'Operating System :: Microsoft',

      ],
      include_package_data=True,
      )
