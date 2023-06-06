# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 29.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""

import numpy as np
from numba import jit
from scipy.stats import gumbel_r, norm

from var.auxiliary import array_like, number

__all__ = ["historic", "parametric", "monte_carlo", "monte_carlo_stressed", "garch", "cdar", "cvar"]


@jit(cache=True)
def historic(pnl: array_like, alpha: number) -> array_like:
    """
    Compute the Value at Risk value with the historical computation approach.

    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    pnl : array_like
        An array with the daily profit and losses.
    alpha : number
        A confidence intervals (alpha) for VaR.

    Returns
    -------
    out : array_like
        Value at Risk value at the chosen confidence interval.

    References
    ----------
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)

    """
    var_value = np.percentile(pnl, alpha)
    cvar_value = np.mean(pnl[pnl <= var_value])
    data = np.array([var_value, cvar_value])

    return data


@jit(cache=True)
def parametric(pnl: array_like, alpha: number, daily_std: number) -> array_like:
    """
    Compute the Value at Risk value with the parametric computation approach.

    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
    and variance of the returns series, assuming normal distribution.

    Parameters
    ----------
    pnl : array_like
        An array with the daily profit and losses.
    alpha : number
        A confidence intervals (alpha) for VaR.
    daily_std : float
        Daily Standard Deviation of the portfolio.

    Returns
    -------
    out : array_like
        Value at Risk value at the chosen confidence interval.

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    """
    z_values = norm.ppf(alpha / 100)

    var_value = np.mean(pnl) + z_values * daily_std
    cvar_value = np.mean(pnl[pnl <= var_value])
    data = np.array([var_value, cvar_value])

    return data


@jit(cache=True)
def monte_carlo(pnl: array_like, alpha: number) -> array_like:
    """
    Compute the Value at Risk value with a Monte Carlo simulation.

    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.

    Parameters
    ---------
    pnl : array_like
        An array with the daily profit and losses.
    alpha : number
        A confidence intervals (alpha) for VaR.

    Returns
    -------
    out : array_like
        Value at Risk value at the chosen confidence interval.

    References
    ----------
    [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
    [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
    [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
    """

    PnL_list = np.random.normal(np.mean(pnl), np.std(pnl), 100000)

    var_value = np.percentile(PnL_list, alpha)
    pnl_frame = PnL_list[PnL_list <= var_value]
    cvar_value = np.nanmean(pnl_frame)
    data = np.array([var_value, cvar_value])

    return data


def monte_carlo_stressed(pnl: array_like, alpha: number) -> array_like:
    """
    Compute the Value at Risk value with a Stressed Monte Carlo simulation.

    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.

    The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
    distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
    extreme value distribution, log-Weibull and Gompertz distributions.

    Parameters
    ---------
    pnl : array_like
        An array with the daily profit and losses.
    alpha : number
        A confidence intervals (alpha) for VaR.

    Returns
    -------
    out : array_like
        Value at Risk value at the chosen confidence interval.

    References
    ----------
    [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
    [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
    [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
    """

    loc, scale = gumbel_r.fit(pnl)
    pnl_list = np.random.gumbel(loc, scale, 100000)

    var_value = np.percentile(pnl_list, alpha)
    pnl_frame = pnl_list[pnl_list <= var_value]
    cvar_value = np.nanmean(pnl_frame)
    data = np.array([var_value, cvar_value])

    return data

@jit(cache=True)
def cvar(pnl: array_like, var_value: number) -> number:
    """
    Compute the Conditional Value at Risk value.

    Parameters
    ----------
    pnl : array_like
        An array with the daily profit and losses.
    var_value : number
        Value at Risk.

    Returns
    -------
    number
        Conditional Value at Risk value.
    """
    cvar_value = np.nanmean(pnl[pnl <= var_value])
    return cvar_value


@jit(cache=True)
def cdar(pnl: array_like, alpha: number, comp: bool = True) -> number:
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of a returns series.

    Parameters
    ----------
    pnl : array_like
        A DataFrame with the daily profit and losses.
    alpha : number
        A list confidence intervals (alpha values) for VaR.
    comp : bool, optional
        If True (Default) use the compounded cumulative returns.

    Returns
    -------
    out : number
        A list object with Value at Risk values at different confidence intervals.
    """
    a = np.array(pnl, ndmin=2)
    if comp:
        prices = 1 + np.insert(a, 0, 0, axis=0)
        NAV = np.cumprod(prices, axis=0).flatten()
        DD = list()

        peak = -99999
        for nav in np.nditer(NAV):
            peak = nav if nav > peak else peak
            DD.append(-(peak - nav) / peak)
        del DD[0]
    else:
        prices = np.insert(a, 0, 1, axis=0)
        NAV = np.cumsum(np.array(prices), axis=0)
        DD = list()

        peak = -99999
        for nav in np.nditer(NAV):
            peak = nav if nav > peak else peak
            DD.append(-(peak - nav))
        del DD[0]

    sorted_DD = np.sort(np.array(DD), axis=0)
    index = int(np.ceil(alpha * len(sorted_DD)) - 1)

    sum_var = 0
    for i in range(index + 1):
        sum_var = sum_var + sorted_DD[i] - sorted_DD[index]
    value = sorted_DD[index] - sum_var / (alpha * len(sorted_DD))

    return value


__METHODS__ = {"h": historic,
               "p": parametric,
               "mc": monte_carlo,
               "smc": monte_carlo_stressed}
