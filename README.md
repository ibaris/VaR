<div align="center">
  <p>
    <a href='https://www.freepik.com/vectors/business'>
      <img src="./resources/logo/var_logo.png" width="700" height="400">
    </a>
  </p>

<h4 align="center">Value-at-Risk</h4>

<p align="center">
  <a href="https://media0.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy.gif">
    <img src="https://forthebadge.com/images/badges/contains-cat-gifs.svg"
         alt="Gitter">
  </a>
  <a href="https://i.makeagif.com/media/10-25-2015/oWi1M-.gif">
    <img src="http://forthebadge.com/images/badges/mom-made-pizza-rolls.svg">
  </a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#dependencies">Dependencies</a> •
</p>
</div>

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
- Parametric GARCH

methods.

# Examples

In this example we will show all the key features of the `var` package. At first we will import all
necessary packages.

```python
from var import VaR, load_data
import numpy as np
```

To quickly test and demonstrate the functions, the package includes a function named `load_data`, which by default
includes the daily returns of stocks `TSLA`, `AAPL` and `NFLX`.

```python
data = load_data()
print(data)
```

```console
              NFLX      AAPL     TSLA
Date
2016-06-28  0.0056  0.001725 -0.00020
2016-06-29  0.0139  0.001075  0.01012
2016-06-30  0.0057  0.002900 -0.00138
2016-07-01  0.0167  0.001000  0.02072
2016-07-05  0.0271 -0.001000  0.00850
            ...       ...      ...
2021-06-21 -0.0464  0.020000 -0.03650
2021-06-22  0.1028  0.018500  0.05460
2021-06-23  0.0426 -0.000700  0.24570
2021-06-24  0.0010 -0.010400  0.04830
2021-06-25 -0.0177 -0.003500 -0.17710
[1258 rows x 3 columns]
```

The only important thing in the data preparation is that the columns contain the individual
positions of the portfolio, and the rows the date. Another important point is that the
column "Date" should be defined as an index, and it must also be formatted as a date.

Now we can define some weights for the individual positions and initialize the `VaR` class:

```python
weights = np.array([0.40, 0.50, 0.10])

var = VaR(data, weights)
```

The standard confidence is at `0.01`. Individual confidences can be defined with the parameter
`alpha`:

```python
var = VaR(data, weights, alpha=0.05)
var
```

```console
<VaR - μ: 0.05%, σ: 3.5096%, Portfolio σ: 3.511%>
```

The `repr` of the class provides the following information:

- μ : The mean return of the portfolio.
- σ : The unweighted volatility of the portfolio.
- Portfolio σ : The weighted volatility of the portfolio.

You can summarize the results of the different methods with the method:

```python
var.summary().round(2)  # or use print(var)
```

```console
                      VaR(0.99)  CVaR(0.99)  ES(0.99)
2022-04-27
Parametric                -0.13       -0.15       -0.64
Historical                -0.20       -0.21       -0.64
Monte Carlo               -0.13       -0.14       -0.64
Stressed Monte Carlo      -0.16       -0.17       -0.64
```

You can access the different VaR methods by using the methods:

```python
var.historic()
```

```console
            VaR(0.99)  CVaR(0.99)
2021-06-25  -0.203479   -0.211246
```

```python
var.parametric()
```

```console
            VaR(0.99)  CVaR(0.99)
2021-06-25  -0.130036   -0.149269
```

```python
var.monte_carlo()
```

```console
            VaR(0.99)  CVaR(0.99)
2021-06-25  -0.132929    -0.13989
```

```python
var.monte_carlo(stressed=True)
```

```console
            VaR(0.99)  CVaR(0.99)
2021-06-25  -0.159149   -0.162875
```

Or access to the Conditional Drawdown at Risk with:

```python
var.es()
```

```console
           ES(0.99)
2021-06-25  -0.636892
```

## Fitting Distribution

_Thank you @Osman Mahmud Kim for the hint!_

By default the `VaR` class chooses a normal distribution of the data. You can override this with another distribution with:

```python
var = VaR(data, weights, alpha=0.05, distribution="t")
var
```

```console
<VaR - μ: 0.06%, σ: 3.4461%, Portfolio σ: 3.4475%>
```

If the distribution is `norm` or `t`, the mean and the standard deviation of the returns are used. The degree of freedom of the `t` distribution is set to `len(returns) - 1`. You can define your own parameter with additional keyword arguments. Lets say, you want a `f` distribution with parameter `dfn, dfd`

```python
var = VaR(data, weights, alpha=0.05, distribution="f", dfn=10000, dfd=500)
var
```

```console
<VaR - μ: 0.06%, σ: 3.4461%, Portfolio σ: 3.4475%>
```

You can also refine the distribution or fit new distributions with the method

```python
var.fit_distributions(include_other=True, verbose=True)
```

```console
Fitting 13 distributions: 100%|██████████| 13/13 [00:00<00:00, 15.33it/s]


Best fits:
----------
         sumsquare_error         aic          bic  kl_div  ks_statistic  \
t             156.808460  180.121004 -2598.062910     inf      0.032883
lognorm       955.113977  672.331106  -325.101442     inf      0.126492
f             957.581920  671.103512  -314.717779     inf      0.129057
norm          957.604766  666.417186  -328.962322     inf      0.127633
gamma         985.488639  655.557001  -285.717340     inf      0.138701

            ks_pvalue
t        1.287770e-01
lognorm  5.266286e-18
f        9.986222e-19
norm     2.523508e-18
gamma    1.421048e-21


Best fit:
---------
Distribution t with parameters {'df': 1.6718900963531307, 'loc': 0.0011873373129386235, 'scale': 0.014747352151602807}
```

You can also define only one distribution with

```python
var.fit_distributions(distribution="gamma")
```

## PELVE

_Thank you @Osman Mahmud Kim for the hint!_

[PELVE](https://www.sciencedirect.com/science/article/abs/pii/S0304407622000380) is essentially a ratio or a multiplier. It tells us how to adjust the confidence level when switching from VaR to ES so that we get an equivalent measure of risk. The formula for calculating PELVE is ES*{1−cɛ}(X)=V aR*{1−ɛ}(X), where ε is a small number close to 0, X is a loss random variable, and c is the PELVE.

The idea here is to answer the question: if we replace VaR with ES in our risk models, how will that affect our estimated capital requirements? Will we need more capital to cover potential losses, or less?

The passage provides an example with ε set at 0.01. If the calculated PELVE is more than 2.5, that means using ES at a 97.5% confidence level (ES*{0.975}) will estimate a higher risk value than using VaR at a 99% confidence level (VaR*{0.99}). In other words, switching to ES would mean we'd need more capital to cover potential losses.

On the other hand, if the PELVE is less than 2.5, that means using ES at a 97.5% confidence level will estimate a lower risk value than using VaR at a 99% confidence level. That means switching to ES would indicate that we need less capital to cover potential losses.

In simple terms, PELVE is a tool to help decide what confidence level to use when switching from VaR to ES, in a way that keeps the estimated level of risk (and hence the amount of capital needed) the same.

In the current version you can calculate the PELVE with the historical or the parametric method:

```python
var.compute_pelve(method="p")
```

```console
(array([8.89189276]), 3.610864030825778e-05)
```

The first value is the PELVE and the second value is the optimization error.

## Backtest

You can backtest the accuracy of each method with the method `backtest` and the method keys:

- 'h': VaR calculated with the historical method,
- 'p': VaR calculated with the parametric method,
- 'mc': VaR calculated with the monte carlo method,
- 'smc': VaR calculated with the stressed monte carlo method,

```python
bth = var.backtest(method='h')
```

```console
Backtest: Historic Method: 100%|██████████| 1008/1008 [00:03<00:00, 332.53it/s]
```

Evaluate the backtest results with the method `evalutate`

```python
var.evaluate(bth)
```

```console
           Amount   Percent Mean Deviation STD Deviation Min Deviation  \
VaR(0.99)      10  0.009921      -0.023765      0.028671     -0.003515
CVaR(0.99)     10  0.009921      -0.023407      0.028545     -0.003382
CDaR(0.99)      6  0.005952      -0.017702       0.02285     -0.001713

           Max Deviation
VaR(0.99)      -0.099554
CVaR(0.99)     -0.098803
CDaR(0.99)     -0.060682
```

The table contains the following information:

- Amount : Total amount of exceptions.
- Percent : Total amount of exceptions in relative to all observations (multiply this by 100 to obtain
  the total amount of exceptions in percent).
- Mean Deviation : The mean value of the exceptions.
- STD Deviation : The standard deviation of the exceptions.
- Min Deviation : The wort overestimation of the value.
- Max Deviation : The worst underestimation of the value.

## Plot Backtest

Plot the backtest results via:

```python
var.var_plot(bth)
```

![img.png](resources/imgs/img.png)

```python
var.cvar_plot(bth)
```

![img.png](resources/imgs/img2.png)

```python
var.cdar_plot(bth)
```

![img.png](resources/imgs/img3.png)

# Installation

There are currently different methods to install `var`.

## Using pip

The `var` package is provided on pip. You can install it with::

    pip install var

## Standard Python

You can also download the source code package from this repository or from pip. Unpack the file you obtained into some directory (
it can be a temporary directory) and then run::

    python setup.py install

# Dependencies

- Python: Python 3.7
- Packages: numpy, pandas, scipy, matplotlib, tqdm, seaborn, numba
