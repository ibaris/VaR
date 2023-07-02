# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 29.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import warnings

import numpy as np
from arch import arch_model
# from numba import jit
from scipy import stats

from var.auxiliary import array_like

__all__ = ["historic", "parametric", "monte_carlo", "garch"]

# @jit(cache=True)
def calculate_expected_shortfall(pnl: array_like, var: array_like) -> np.ndarray:
    """ Compute the expected Shortfall

    Parameters
    ----------
    pnl : array_like
        Profit and Loss values.
    var : array_like
        Value at Risk values.

    Returns
    -------
    np.ndarray
        Expected Shortfall
    """
    es_values = np.zeros_like(var)

    for i, item in enumerate(var):
        # Find the returns which are less than VaR (i.e., in the tail)
        tail_returns = pnl[pnl < item]

        if len(tail_returns) == 0:
            warnings.warn("VaR is too high. No returns were found below the VaR level. "
                          "Please check the inputs or increase the VaR level.")
            es_values[i] = item
            continue

        # Calculate the mean of the tail returns
        es_values[i] = np.mean(tail_returns)

    return es_values

# @jit(cache=True)
def compute_cdar(pnl: array_like, var: array_like) -> np.ndarray:
    """Compute the Drawdown of a portfolio

    Parameters
    ----------
    pnl : array_like
        Profit and Loss values.

    Returns
    -------
    np.ndarray
        Drawdowns
    """
    # Compute the drawdowns
    running_max = np.maximum.accumulate(pnl)
    drawdowns = running_max - pnl

    drawdown_values = np.zeros_like(var)

    for i, item in enumerate(var):
        drawdown_values[i] = np.nanmean(drawdowns[drawdowns > item])

    drawdown_values = np.nan_to_num(drawdown_values)

    return drawdown_values


def historic(pnl: array_like, alpha: array_like, **kwargs) -> np.ndarray:
    """
    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    pnl : array_like
        A DataFrame with the daily profit and losses.
    alpha : array_like
        A list significance levels (alpha values) for VaR.

    Returns
    -------
    out : np.ndarray
        A list object with Value at Risk values at different significance levels.

    References
    ----------
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)

    """
    confidence_level = 1 - alpha

    var_values = np.percentile(pnl, 100 - (confidence_level * 100), interpolation="lower")
    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    cdar_values = compute_cdar(pnl=pnl, var=var_values)

    data = np.concatenate((var_values, es_values, cdar_values))

    return data


def parametric(pnl: array_like, alpha: array_like, daily_std: float, ppf: callable = stats.norm.ppf, **kwargs) -> np.ndarray:
    """
    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
    and variance of the returns series, assuming normal distribution.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : np.ndarray
        A list significance levels (alpha values) for VaR.
    daily_std : float
        Daily Standard Deviation of the portfolio.
    ppf : callable
        Percent point function (inverse of cdf — percentiles). Default is `stats.norm.ppf`.

    Returns
    -------
    out : np.ndarray
        A list object with Value at Risk values at different significance levels.

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    """
    # See [here](https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf)
    # If you're interested in a 99% confidence interval (one tail), you will feed ppf with 0.99.
    # If it's a two-tailed test, you would provide 0.995 for the upper tail and 0.005 for the lower
    # tail (for a 99% confidence interval).
    z_values = ppf(alpha, **kwargs)

    var_values = np.mean(pnl) + z_values * daily_std
    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    cdar_values = compute_cdar(pnl=pnl, var=var_values)

    data = np.concatenate((var_values, es_values, cdar_values))

    return data


def monte_carlo(
    pnl: array_like,
    alpha: array_like,
    rvs: callable = stats.norm.rvs,
    **kwargs
) -> np.ndarray:
    """
    The Monte Carlo Method involves developing a model for future stock price returns and running multiple
    hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
    generates trials, but by itself does not tell us anything about the underlying methodology.
    
    The Stressed Monte Carlo Method uses the Gumel distribution (gummel_r) to generate the random trials. 
    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also
    related to the extreme value distribution, log-Weibull and Gompertz distributions.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list significance levels (alpha values) for VaR.
    rvs : callable
        Random variates of given type. Default is `stats.norm.rvs`.
    
    Returns
    -------
    out : list
        A list object with Value at Risk values at different significance levels.

    Notes
    -----
    The Stressed Monte Carlo Method uses the Gumel distribution ('gumbel_r') to generate the random trials. 
    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also 
    related to the extreme value distribution, log-Weibull and Gompertz distributions.

    References
    ----------
    [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
    [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
    [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
    """
    confidence_level = 1 - alpha
    n_simulations = 5000

    # Run the Monte Carlo simulation: generate random numbers from a fitted distribution
    simulated_returns = rvs(size=n_simulations, **kwargs)

    # Sort the simulated returns in ascending order
    simulated_returns = np.sort(simulated_returns)

    # Compute the VaR at the desired confidence level
    var_values = [-simulated_returns[int(n_simulations * item)] for item in confidence_level]

    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    cdar_values = compute_cdar(pnl=pnl, var=var_values)

    data = np.concatenate((var_values, es_values, cdar_values))

    return data


def garch(pnl, alpha, **kwargs):
    """
    This method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
    model.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list significance levels (alpha values) for VaR.

    Returns
    -------
    out : list
        A list object with Value at Risk values at different significance levels.

    References
    ----------
    [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Specify the GARCH model
        model = arch_model(pnl, vol='Garch', p=1, q=1)

        # Fit the model
        model_fit = model.fit(disp='off')

        # Compute conditional standard deviations from the model
        conditional_volatilities = model_fit.conditional_volatility

    # Compute VaR at the desired confidence level (e.g., 99%)
    var_values = [(stats.norm.ppf(item) * conditional_volatilities)[-1] for item in alpha]

    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    cdar_values = compute_cdar(pnl=pnl, var=var_values)

    data = np.concatenate((var_values, es_values, cdar_values))

    return data
