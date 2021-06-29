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
from scipy.stats import gumbel_r
from scipy.stats import norm
import pandas as pd

__all__ = ["historic", "parametric", "monte_carlo", "garch"]


def historic(daily_pnl, alpha):
    """
    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    daily_pnl : pd.DataFrame
        A DataFrame with the daily profit and losses.
    alpha : list or None
        A list confidence intervals (alpha values) for VaR. If None, the default values are [5%, 2.5%, 1%].

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
    alpha : list or None
        A list confidence intervals (alpha values) for VaR. If None, the default values are [5%, 2.5%, 1%].
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
    alpha : list or None
        A list confidence intervals (alpha values) for VaR. If None, the default values are [5%, 2.5%, 1%].
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
        PnL_list = np.random.normal(np.mean(daily_pnl.values), np.std(daily_pnl.values), 500000)
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
    alpha : list or None
        A list confidence intervals (alpha values) for VaR. If None, the default values are [5%, 2.5%, 1%].

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

    q = am.distribution.ppf(alpha / 100, res.params[-2:])
    value_at_risk = forecasts.mean.values + np.sqrt(forecasts.variance).values * q[None, :]

    var_values = [value_at_risk[[-1, ]][0][0], value_at_risk[[-1, ]][0][1], value_at_risk[[-1, ]][0][2]]
    cvar_values = [np.mean(daily_pnl[daily_pnl <= item]) for item in var_values]
    cvar_values = [var_values[-1] if math.isnan(item) else item for item in cvar_values]

    data = np.append(var_values, cvar_values).flatten()

    return data
