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
import pandas as pd
from arch import arch_model
from scipy.stats import gumbel_r
from scipy.stats import norm

__all__ = ["historic", "parametric", "monte_carlo", "garch", "cdar"]


def historic(daily_pnl, alpha):
    """
    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    daily_pnl : pd.DataFrame
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)

    """
    var_values = np.percentile(daily_pnl, alpha)
    cvar_values = [np.mean(daily_pnl[daily_pnl <= item]) for item in var_values]
    data = np.append(var_values, cvar_values).flatten()

    return data


def parametric(daily_pnl, alpha, daily_std):
    """
    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
    and variance of the returns series, assuming normal distribution.

    Parameters
    ----------
    daily_pnl : pd.DataFrame
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.
    daily_std : float
        Daily Standard Deviation of the portfolio.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different confidence intervals.

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    """
    z_values = norm.ppf(alpha / 100)

    var_values = np.mean(daily_pnl.values) + z_values * daily_std
    cvar_values = [np.mean(daily_pnl[daily_pnl <= item]) for item in var_values]
    data = np.append(var_values, cvar_values).flatten()

    return data


def monte_carlo(daily_pnl, alpha, stressed=False):
    """
    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.

    The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
    distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
    extreme value distribution, log-Weibull and Gompertz distributions.

    Parameters
    ---------
    daily_pnl : pd.DataFrame
        A DataFrame with the daily profit and losses.
    alpha : list
        A list confidence intervals (alpha values) for VaR.
    stressed : bool
        Use the Stressed Monte Carlo Method. Default is False.

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

    if not stressed:
        PnL_list = np.random.normal(np.mean(daily_pnl.values), np.std(daily_pnl.values), 100000)
    else:
        block_maxima = daily_pnl.resample('W').min().dropna()
        loc, scale = gumbel_r.fit(block_maxima)
        PnL_list = np.random.gumbel(loc, scale, 100000)

    var_values = np.percentile(PnL_list, alpha)
    PnL_frame = [PnL_list[PnL_list <= item] for item in var_values]
    cvar_values = [np.nanmean(item) for item in PnL_frame]
    data = np.append(var_values, cvar_values).flatten()

    return data


def garch(daily_pnl, alpha):
    """
    This method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
    model.

    Parameters
    ----------
    daily_pnl : pd.DataFrame
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
        am = arch_model(daily_pnl, vol='Garch', p=1, o=0, q=1, dist='skewt')
        res = am.fit(disp='off')
        forecasts = res.forecast(reindex=True)

    q = am.distribution.ppf(alpha, res.params[-2:])
    value_at_risk = forecasts.mean.values + np.sqrt(forecasts.variance).values * q[None, :]

    var_values = [float(value_at_risk[[-1, ]][0][0]), float(value_at_risk[[-1, ]][0][1]),
                  float(value_at_risk[[-1, ]][0][2])]

    cvar_values = [np.mean(daily_pnl[daily_pnl <= item]) for item in var_values]
    cvar_values = [var_values[-1] if math.isnan(item) else item for item in cvar_values]

    data = np.append(var_values, cvar_values).flatten()

    return data


def cdar(daily_pnl, alpha, comp=True):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of a returns series.

    Parameters
    ----------
    daily_pnl : pd.DataFrame
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
    a = np.array(daily_pnl, ndmin=2)
    if comp:
        prices = 1 + np.insert(np.array(a), 0, 0, axis=0)
        NAV = np.cumprod(prices, axis=0)
        DD = list()

        peak = -99999
        for i in NAV:
            peak = i if i > peak else peak
            DD.append(-(peak - i) / peak)
        del DD[0]
    else:
        prices = np.insert(np.array(a), 0, 1, axis=0)
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
        value.append(np.array(value_tmp).item())

    value.extend(value)

    return value
