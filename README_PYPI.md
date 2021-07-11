# Introduction

"The search for appropriate risk measuring methodologies has been followed by increased financial uncertainty worldwide. Financial
turmoil and the increased volatility of financial markets have induced the design and development of more sophisticated tools for
measuring and forecasting risk. The most well known risk measure is value at risk (VaR), which is defined as the maximum loss over
a targeted horizon for a given level of confidence. In other words, it is an estimation of the tails of the empirical distribution
of financial losses. It can be used in all types of financial risk
measurement" ([Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)).

In addition to Value at Risk, the package includes Conditional Value at Risk (Expected Shortfall or CVaR) and Conditional Drawdown
at Risk (CDaR).

# Key Features

Calculate, Backtest and Plot the

- Value at Risk,
- Conditional Value at Risk,
- Conditional Drawdown at Risk, 
  
with different methods, such that:
- Historical
- Parametric
- Monte Carlo
- Stressed Monte Carlo
- Parametric GARCH 
  
methods.

# Examples
For examples see [here]('https://github.com/ibaris/VaR')

# Installation

There are currently different methods to install `var`.

### Using pip

The ` var ` package is provided on pip. You can install it with::

    pip install var

### Standard Python

You can also download the source code package from this repository or from pip. Unpack the file you obtained into some directory (
it can be a temporary directory) and then run::

    python setup.py install

# Dependencies

* Python: Python 3.7
* Packages: numpy, pandas, arch, scipy, matplotlib, tqdm, seaborn, numba
