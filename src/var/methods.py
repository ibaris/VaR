"""
VaR Methods Module
==================
*Created on 29.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

This module provides a suite of methods for calculating Value at Risk (VaR) and related risk metrics for financial portfolios. It includes
implementations of several standard approaches to VaR estimation, including historical, parametric, Monte Carlo, and GARCH-based methods.
Additionally, it provides functions to compute Expected Shortfall (ES) and Conditional Drawdown at Risk (CDaR), which are important
complementary risk measures.

Functions
---------
- calculate_expected_shortfall: Computes the expected shortfall given profit and loss (PnL) data and VaR values.
- compute_drawdown: Calculates the conditional drawdown at risk for a portfolio.
- historic: Estimates VaR using the historical simulation method.
- parametric: Estimates VaR using the parametric (variance-covariance) method, assuming a specified distribution.
- monte_carlo: Estimates VaR using Monte Carlo simulation, with support for custom distributions.
- garch: Estimates VaR using a GARCH(1,1) model for volatility forecasting.

All methods return concatenated arrays of VaR, Expected Shortfall, and CDaR values for specified significance levels. The module is
designed for extensibility and integration with portfolio risk management workflows.

"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from arch import arch_model
from scipy import stats
from tqdm import tqdm

from var.auxiliary import RiskResults, array_like

__all__ = ["historic", "parametric", "monte_carlo", "garch"]


def calculate_expected_shortfall(pnl: array_like, var: array_like) -> np.ndarray:
    """
    Compute the expected Shortfall

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
    masks = [pnl < var[:, [i]] for i in range(var.shape[-1])]
    tails = np.where(masks, pnl, np.nan)
    es_values = np.nanmean(tails, axis=-1).T

    mask = np.isnan(es_values)
    es_values[mask] = var[mask]

    return es_values


def compute_drawdown(pnl: array_like, var: array_like, axis: int = 0) -> np.ndarray:
    """
    Compute the Drawdown of a portfolio

    Parameters
    ----------
    pnl : array_like
        Profit and Loss values.
    var :array_like
        Value at Risk array.

    Returns
    -------
    np.ndarray
        Drawdowns
    """
    # Compute the drawdowns
    running_max = np.maximum.accumulate(pnl, axis=axis)
    drawdowns = running_max - pnl

    masks = [drawdowns > var[:, [i]] for i in range(var.shape[-1])]
    tails = np.where(masks, drawdowns, np.nan)
    dd_values = np.nanmean(tails, axis=-1)

    return -dd_values.T


def historic(pnl: array_like, alpha: array_like, axis: int | tuple[int] | None = None, **kwargs) -> np.ndarray:
    """
    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    pnl : array_like
        A DataFrame with the daily profit and losses.
    alpha : array_like
        A list significance levels (alpha values) for VaR.
    axis : int, tuple of int, None, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.
    **kwargs : dict
        Additional keyword Arguments that are not used in this function.

    Returns
    -------
    out : np.ndarray
        A list object with Value at Risk values at different significance levels.

    References
    ----------
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)

    """
    confidence_level = 1 - alpha

    var_values = np.atleast_2d(
        np.percentile(
            pnl,
            100 - (confidence_level * 100),
            method="lower",
            axis=axis,
        ),
    )  # Shape: (T, Alpha)

    if axis is not None:
        var_values = var_values.T  # Shape: (T, Alpha)

    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    dd_values = compute_drawdown(pnl=pnl, var=var_values, axis=1)

    results = RiskResults(alpha=alpha, var=var_values, es=es_values, dd=dd_values, name="Historic")

    return results


def parametric(
    pnl: array_like,
    alpha: array_like,
    std: float,
    ppf: callable = stats.norm.ppf,
    axis: int | tuple[int] | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """
    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
    and variance of the returns series, assuming a given distribution.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : np.ndarray
        A list significance levels (alpha values) for VaR.
    std : float
        Standard Deviation of the portfolio.
    ppf : callable
        Percent point function (inverse of cdf — percentiles). Default is `stats.norm.ppf`.
    axis : int, tuple of int, None, optional
        Axis or axes along which the means are computed. The
        default is to compute the mean along a flattened
        version of the array.
    **kwargs : dict
        Further keyword arguments for the ppf function.

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

    z_values = np.atleast_2d(ppf(alpha, **kwargs)).T

    var_values = np.atleast_2d(np.mean(pnl, axis=axis)).T + (z_values * std).T
    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    dd_values = compute_drawdown(pnl=pnl, var=var_values, axis=1)

    results = RiskResults(alpha=alpha, var=var_values, es=es_values, dd=dd_values, name="Parametric")

    return results


def monte_carlo(
    pnl: array_like,
    alpha: array_like,
    rvs: callable = stats.norm.rvs,
    axis: int | tuple[int] | None = None,
    **kwargs: dict,
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
    axis : int, tuple of int, None, optional
        Axis or axes along which the array is sorted. The
        default is to sort the array along a flattened
        version of the array.
    **kwargs : dict
        Further keyword arguments for the rvs function.

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
    confidence_level = alpha
    n_simulations = 1000

    # Run the Monte Carlo simulation: generate random numbers from a fitted distribution
    simulated_returns = np.atleast_2d(rvs(size=(pnl.shape[0], n_simulations), **kwargs))

    # Sort the simulated returns in ascending order
    simulated_returns = np.atleast_2d(np.sort(simulated_returns, axis=axis))

    # Compute the VaR at the desired confidence level
    var_values = np.array([simulated_returns[:, int(n_simulations * item)] for item in confidence_level]).T

    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    dd_values = compute_drawdown(pnl=pnl, var=var_values)

    results = RiskResults(alpha=alpha, var=var_values, es=es_values, dd=dd_values, name="Monte Carlo")

    return results


def garch(
    pnl: array_like,
    alpha: array_like,
    dist: Literal["normal", "gaussian", "t", "studentst", "ged", "generalized error"] = "normal",
    **kwargs: dict,
):
    """
    The GARCH method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
    model.

    Parameters
    ----------
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    alpha : list
        A list significance levels (alpha values) for VaR.
    ppf : callable
        Percent point function (inverse of cdf — percentiles). Default is `stats.norm.ppf`.
    axis : int, tuple of int, None, optional
        Axis or axes along which the array is fitted to the GARCH model. The
        default is to fit the array along a flattened
        version of the array.
    **kwargs : dict
        Further keyword arguments for the ppf function.

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

        args = ()
        if dist in ["t", "studentst", "ged", "generalized error"]:
            args = min(pnl.shape[-1] - 1, 500)
            args = (args,)

        var_values = np.zeros((pnl.shape[0], len(alpha)))

        for i, data in enumerate(tqdm(pnl, desc="Fitting GARCH Model", leave=False)):
            # Specify the GARCH model
            model = arch_model(data, vol="Garch", p=1, q=1, dist=dist)

            # Fit the model
            model_fit = model.fit(disp="off")

            # Compute conditional standard deviations from the model
            conditional_volatility = model_fit.conditional_volatility

            # Compute VaR at the desired confidence level (e.g., 99%)
            var_values[i] = [(model.distribution.ppf(item, *args, **kwargs) * conditional_volatility)[-1] for item in alpha]

    es_values = calculate_expected_shortfall(pnl=pnl, var=var_values)
    dd_values = compute_drawdown(pnl=pnl, var=var_values)

    results = RiskResults(alpha=alpha, var=var_values, es=es_values, dd=dd_values, name="Monte Carlo")

    return results
