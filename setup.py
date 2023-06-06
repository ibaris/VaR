# -*- coding: utf-8 -*-
"""
Setup of VaR Package
====================
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import io
import os
import platform
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext
from setuptools import find_packages
from setuptools import setup

long_description = """"The search for appropriate risk measuring methodologies has been followed by increased financial uncertainty worldwide. Financial
turmoil and the increased volatility of financial markets have induced the design and development of more sophisticated tools for
measuring and forecasting risk. The most well known risk measure is value at risk (VaR), which is defined as the maximum loss over
a targeted horizon for a given level of confidence. In other words, it is an estimation of the tails of the empirical distribution
of financial losses. It can be used in all types of financial risk
measurement" ([Julija Cerović Smolovic, 2017](https://doi.org/10.1080/1331677X.2017.1305773)).

In addition to Value at Risk, the package includes Conditional Value at Risk (Expected Shortfall or CVaR) and Conditional Drawdown
at Risk (CDaR).
"""


# ------------------------------------------------------------------------------------------------------------
# Environmental Functions
# ------------------------------------------------------------------------------------------------------------
with open('requirements.txt', encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name='var',
    version="2023.6.1",
    description='Different Methods to Estimate the Value-at-Risk of a portfolio.',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    author="Ismail Baris",
    maintainer='Ismail Baris',
    author_email='i.baris@outlook.de',
    url='https://github.com/ibaris/VaR',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Office/Business :: Financial :: Investment',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft',
    ],
)
