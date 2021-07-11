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
- Stressed Monte Carlo
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
positions of the portfolio, and the rows the date.  Another important point is that the 
column "Date" should be defined as an index, and it must also be formatted as a date.

Now we can define some weights for the individual positions and initialize the `VaR` class:
```python
weights = np.array([0.40, 0.50, 0.10])

var = VaR(data, weights)
```
The standard confidence is at `0.05, 0.025, 0.01`. Individual confidences can be defined with the parameter
`alpha`:
```python
var = VaR(data, weights, alpha=[0.1, 0.05, 0.01])
var
```
```console
<VaR - μ: 0.05%, σ: 3.5096%, Portfolio σ: 3.511%>
```
The `repr` of the class provides the following information:
* μ : The mean return of the portfolio.
* σ : The unweighted volatility of the portfolio.
* Portfolio σ : The weighted volatility of the portfolio.

You can summarize the results of the different methods with the method:
```python
var.summary()  # or use print(var)
```
```console
                      VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-07-11                                                                      
Parametric            -0.107960  -0.114992  -0.130036   -0.133072   -0.142010   
Historical            -0.148293  -0.172413  -0.203479   -0.180353   -0.211246   
Monte Carlo           -0.108710  -0.115016  -0.129083   -0.117886   -0.123955   
Stressed Monte Carlo  -0.146094  -0.150056  -0.158214   -0.151642   -0.155467   
GARCH                 -0.037688  -0.263218  -0.442813   -0.067248   -0.442813   
                      CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-07-11                                                            
Parametric             -0.149269   -0.593583   -0.628195   -0.658899  
Historical             -0.211246   -0.593583   -0.628195   -0.658899  
Monte Carlo            -0.135459   -0.593583   -0.628195   -0.658899  
Stressed Monte Carlo   -0.162193   -0.999779   -0.999879   -0.999910  
GARCH                  -0.442813   -0.593583   -0.628195   -0.658899  
```

You can access the different VaR methods by using the methods:
```python
var.historic()
```
```console
            VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-06-25  -0.148293  -0.172413  -0.203479   -0.180353   -0.211246   
            CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-06-25   -0.211246   -0.593583   -0.628195   -0.658899 
```
```python
var.parametric()
```
```console
            VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-06-25   -0.10796  -0.114992  -0.130036   -0.133072    -0.14201   
            CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-06-25   -0.149269   -0.593583   -0.628195   -0.658899  
```
```python
var.monte_carlo()
```
```console
            VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-06-25  -0.107858  -0.113826  -0.127765   -0.116556   -0.122468   
            CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-06-25    -0.13268   -0.593583   -0.628195   -0.658899  
```
```python
var.monte_carlo(stressed=True)
```
```console
            VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-06-25   -0.14644  -0.150529  -0.161256   -0.152076   -0.155756   
            CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-06-25   -0.164127   -0.999779   -0.999879    -0.99991  
```
```python
var.garch(stressed=True)
```
```console
            VaR(90.0)  VaR(95.0)  VaR(99.0)  CVaR(90.0)  CVaR(95.0)  \
2021-06-25  -0.037688  -0.263218  -0.442813   -0.067248   -0.442813   
            CVaR(99.0)  CDaR(90.0)  CDaR(95.0)  CDaR(99.0)  
2021-06-25   -0.442813   -0.593583   -0.628195   -0.658899    
```
### Backtest
You can backtest the accuracy of each method with the method `backtest` and the method keys:
* 'h': VaR calculated with the historical method,
* 'p': VaR calculated with the parametric method,
* 'mc': VaR calculated with the monte carlo method,
* 'smv': VaR calculated with the stressed monte carlo method,
* 'g': VaR calculated with the garch method.

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
Observations   1008         1              0             0             0   
VaR(90.0)        10  0.009921      -0.025198      0.027696     -0.004046   
VaR(99.0)        10  0.009921      -0.025198      0.029195     -0.004046   
CVaR(90.0)       10  0.009921      -0.023407       0.02708     -0.003382   
CVaR(99.0)       10  0.009921      -0.023407      0.028545     -0.003382   
CDaR(90.0)        5   0.00496      -0.024784      0.020734      -0.00414   
CDaR(99.0)        2  0.001984      -0.043007      0.027027     -0.023896   
             Max Deviation  
Observations             0  
VaR(90.0)        -0.102561  
VaR(99.0)        -0.102561  
CVaR(90.0)       -0.098803  
CVaR(99.0)       -0.098803  
CDaR(90.0)       -0.062118  
CDaR(99.0)       -0.062118
```

The table contains the following information:
* Amount : Total amount of exceptions.
* Percent : Total amount of exceptions in relative to all observations (multiply this by 100 to obtain
  the total amount of exceptions in percent).
* Mean Deviation : The mean value of the exceptions.
* STD Deviation : The standard deviation of the exceptions.
* Min Deviation : The wort overestimation of the value.
* Max Deviation : The worst underestimation of the value.


### Plot Backtest
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
