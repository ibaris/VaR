# -*- coding: utf-8 -*-
"""
Setup of VaR Package
====================
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

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
# Function to read the contents of requirements.txt
def read_requirements():
    """Read requirements.txt, returning a list of dependencies.

    Returns
    -------
    list of str
        List containing the names of the required packages.
    """
    with open('requirements.txt', encoding='utf-8') as file:
        return file.read().splitlines()


setup(
    name='var',
    version="2024.3.0",
    description='Different Methods to Estimate the Value-at-Risk of a portfolio.',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=read_requirements(),  # Required packages
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft',
    ],
)
