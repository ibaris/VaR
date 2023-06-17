# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 29.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import math
import warnings

import numpy as np
from arch import arch_model
from numba import jit
from scipy.stats import gumbel_r
from scipy.stats import norm

__all__ = ["historic", "parametric", "monte_carlo", "monte_carlo_stressed", "garch", "cdar"]


@jit(cache=True)
def historic(pnl, alpha):
    """
    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : np.ndarray
        A list confidence intervals (alpha values) for VaR.

    Returns
    -------
    out : np.ndarray
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)

    """
    var_values = np.percentile(pnl, alpha * 100, interpolation="lower")
    cvar_values = [np.mean(pnl[pnl <= item]) for item in var_values]
    cdar_values = cdar(pnl, alpha)
    data = np.concatenate((var_values, cvar_values, cdar_values))

    return data


@jit(cache=True)
def parametric(pnl, alpha, daily_std):
    """
    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
    and variance of the returns series, assuming normal distribution.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : np.ndarray
        A list confidence intervals (alpha values) for VaR.
    daily_std : float
        Daily Standard Deviation of the portfolio.

    Returns
    -------
    out : np.ndarray
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    """
    # See [here](https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf)
    z_values = norm.ppf((1 - alpha) / 100)

    var_values = np.mean(pnl) + z_values * daily_std
    cvar_values = [np.mean(pnl[pnl <= item]) for item in var_values]
    cdar_values = cdar(pnl, alpha)
    data = np.concatenate((var_values, cvar_values, cdar_values))

    return data


@jit(cache=True)
def monte_carlo(pnl, alpha):
    """
    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.

    The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
    distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
    extreme value distribution, log-Weibull and Gompertz distributions.

    Parameters
    ---------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
    [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
    [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
    """

    PnL_list = np.random.normal(np.mean(pnl), np.std(pnl), 100000)

    var_values = np.percentile(PnL_list, alpha * 100, interpolation="lower")
    pnl_frame = [PnL_list[PnL_list <= item] for item in var_values]
    cvar_values = [np.nanmean(item) for item in pnl_frame]
    cdar_values = cdar(pnl, alpha)

    data = np.concatenate((var_values, cvar_values, cdar_values))

    return data


@jit(cache=True)
def monte_carlo_stressed(pnl, alpha):
    """
    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.

    The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
    distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
    extreme value distribution, log-Weibull and Gompertz distributions.

    Parameters
    ---------
    pnl : np.ndarray
        A DataFrame with the weekly profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
    [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
    [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
    """

    loc, scale = gumbel_r.fit(pnl)
    pnl_list = np.random.gumbel(loc, scale, 100000)

    var_values = np.percentile(pnl_list, alpha * 100, interpolation="lower")
    pnl_frame = [pnl_list[pnl_list <= item] for item in var_values]
    cvar_values = [np.nanmean(item) for item in pnl_frame]
    cdar_values = cdar(pnl, alpha)
    data = np.concatenate((var_values, cvar_values, cdar_values))

    return data


def garch(pnl, alpha):
    """
    This method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
    model.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(pnl, vol='Garch', p=1, o=0, q=1, dist='skewt')
        res = am.fit(disp='off')
        forecasts = res.forecast(reindex=True)

    q = am.distribution.ppf(alpha, res.params[-2:])
    value_at_risk = forecasts.mean.values + np.sqrt(forecasts.variance).values * q[None, :]

    var_values = [float(value_at_risk[[-1, ]][0][0]), float(value_at_risk[[-1, ]][0][1]),
                  float(value_at_risk[[-1, ]][0][2])]

    cvar_values = [np.mean(pnl[pnl <= item]) for item in var_values]
    cvar_values = [var_values[-1] if math.isnan(item) else item for item in cvar_values]

    cdar_values = cdar(pnl, alpha)
    data = np.concatenate((var_values, cvar_values, cdar_values))

    return data


@jit(cache=True)
def cdar(pnl, alpha, comp=True):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of a returns series.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.
    comp : bool
        If True (Default) use the compounded cumulative returns.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.
    """
    a = np.array(pnl, ndmin=2)
    if comp:
        prices = 1 + np.insert(a, 0, 0, axis=0)
        NAV = np.cumprod(prices, axis=0).flatten()
        DD = list()

        peak = -99999
        for i in NAV:
            peak = i if i > peak else peak
            DD.append(-(peak - i) / peak)
        del DD[0]
    else:
        prices = np.insert(a, 0, 1, axis=0)
        NAV = np.cumsum(np.array(prices), axis=0)
        DD = []

        peak = -99999
        for i in NAV:
            peak = i if i > peak else peak
            DD.append(-(peak - i))
        del DD[0]

    sorted_DD = np.sort(np.array(DD), axis=0)
    indices = [int(np.ceil(item * len(sorted_DD)) - 1) for item in alpha]

    value = list()
    for i, index in enumerate(indices):
        sum_var = 0
        for j in range(index + 1):
            sum_var = sum_var + sorted_DD[j] - sorted_DD[index]
        value_tmp = -sorted_DD[index] - sum_var / (alpha[i] * len(sorted_DD))
        value.append(-np.array(value_tmp).item())

    return value


__METHODS__ = {"h": historic,
               "p": parametric,
               "mc": monte_carlo,
               "smc": monte_carlo_stressed,
               "g": garch}
