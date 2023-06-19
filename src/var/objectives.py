# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Objective Functions
===================
*Created on 19.06.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This files contains the objective functions in order to perform optimization tasks.
"""
import numpy as np
from numba import jit
from scipy.stats import norm

from var.auxiliary import array_like

__all__ = ["pelve_parameteric", "pelve_historic"]


def pelve_parameteric(var_value: float, pnl: array_like, daily_std: float):
    """
    This function computes the ES value with a given significance level (alpha) with the parametric method and
    returns the absolute difference to the desired Value at Risk value. This function is used in the optimization
    process to find the optimal ES value for a given VaR value in order to compute the PELVE.

    Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of 
    mean and variance of the returns series, assuming normal distribution.

    Parameters
    ----------
    var_value : float
        Value at Risk (VaR) value at a predetermined significance level.
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.
    daily_std : float
        Daily Standard Deviation of the portfolio.

    Returns
    -------
    out : callable
        A function that returns the absolute difference between the ES value and the desired VaR value. To use
        this function in an optimization process, the function must be called with the desired significance level
        (alpha) as input.

    Examples
    --------
    >>> import numpy as np
    >>> from var import VaR, load_data, objectives
    >>> data = load_data()
    >>> weights = np.array([0.30, 0.60, 0.10])
    >>> var = VaR(data, weights)
    >>> var_value = var.parametric()["VaR(99.0)"].values[0]
    >>> objective = objectives.pelve_parameteric(var_value, var.pnl.values.flatten(), var.info["Portfolio Volatility"])
    >>> objective(0.01)
    0.027561900353827606
    >>> objective(0.075)  # This means, that the ES value at a 92.5% confidence level is approximately the same as the VaR value at 99% confidence level.
    0.00033400209647668766

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    [PELVE: Probability Equivalent Level of VaR and ES](https://www.sciencedirect.com/science/article/pii/S0304407622000380)

    """

    @jit(cache=True)
    def objective(alpha: float) -> float:
        """The objective function that returns the absolute difference between the ES value and the desired 
        VaR value at a inputted significance level alpha.

        Parameters
        ----------
        alpha : float
            Significance level (alpha) for the ES value.

        Returns
        -------
        float
            Absolute difference between the ES value and the desired VaR value at a inputted significance level alpha.

        Examples
        --------
        >>> import numpy as np
        >>> from var import VaR, load_data, objectives
        >>> data = load_data()
        >>> weights = np.array([0.30, 0.60, 0.10])
        >>> var = VaR(data, weights)
        >>> var_value = var.parametric()["VaR(99.0)"].values[0]
        >>> objective = objectives.pelve_parameteric(var_value, var.pnl.values.flatten(), var.info["Portfolio Volatility"])
        >>> objective(0.01)
        0.027561900353827606
        >>> objective(0.075)  # This means, that the ES value at a 92.5% confidence level is approximately the same as the VaR value at 99% confidence level.
        0.00033400209647668766

        References
        ----------
        [Risk.net](https://www.risk.net/definition/value-at-risk-var)
        [PELVE: Probability Equivalent Level of VaR and ES](https://www.sciencedirect.com/science/article/pii/S0304407622000380)

        """
        # See [here](https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf)
        # If you're interested in a 99% confidence interval (one tail), you will feed ppf with 0.99.
        # If it's a two-tailed test, you would provide 0.995 for the upper tail and 0.005 for the lower
        # tail (for a 99% confidence interval).
        z_values = norm.ppf(alpha)

        var_value_es = np.mean(pnl) + z_values * daily_std
        es_value = np.mean(pnl[pnl < var_value_es])

        loss = abs(es_value - var_value)

        return loss

    return objective


def pelve_historic(var_value: float, pnl: array_like):
    """
    This function computes the ES value with a given significance level (alpha) with the historic method and
    returns the absolute difference to the desired Value at Risk value. This function is used in the optimization
    process to find the optimal ES value for a given VaR value in order to compute the PELVE.

    The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
    It then assumes that history will repeat itself, from a risk perspective.

    Parameters
    ----------
    var_value : float
        Value at Risk (VaR) value at a predetermined significance level.
    pnl : np.ndarray
        A DataFrame with the daily profit and losses.

    Returns
    -------
    out : callable
        A function that returns the absolute difference between the ES value and the desired VaR value. To use
        this function in an optimization process, the function must be called with the desired significance level
        (alpha) as input.

    Examples
    --------
    >>> import numpy as np
    >>> from var import VaR, load_data, objectives
    >>> data = load_data()
    >>> weights = np.array([0.30, 0.60, 0.10])
    >>> var = VaR(data, weights)
    >>> var_value = var.historic()["VaR(99.0)"].values[0]
    >>> objective = objectives.pelve_historic(var_value, var.pnl.values.flatten(), var.info["Portfolio Volatility"])
    >>> objective(0.01)
    0.048507266158425294
    >>> objective(0.065)  # This means, that the ES value at a 93.5% confidence level is approximately the same as the VaR value at 99% confidence level.
    0.0008766274547215847

    References
    ----------
    [Risk.net](https://www.risk.net/definition/value-at-risk-var)
    [PELVE: Probability Equivalent Level of VaR and ES](https://www.sciencedirect.com/science/article/pii/S0304407622000380)

    """

    @jit(cache=True)
    def objective(alpha: float) -> float:
        """The objective function that returns the absolute difference between the ES value and the desired 
        VaR value at a inputted significance level alpha.

        Parameters
        ----------
        alpha : float
            Significance level (alpha) for the ES value.

        Returns
        -------
        float
            Absolute difference between the ES value and the desired VaR value at a inputted significance level alpha.

        Examples
        --------
        >>> import numpy as np
        >>> from var import VaR, load_data, objectives
        >>> data = load_data()
        >>> weights = np.array([0.30, 0.60, 0.10])
        >>> var = VaR(data, weights)
        >>> var_value = var.parametric()["VaR(99.0)"].values[0]
        >>> objective = objectives.pelve_historic(var_value, var.pnl.values.flatten(), var.info["Portfolio Volatility"])
        >>> objective(0.01)
        0.048507266158425294
        >>> objective(0.065)  # This means, that the ES value at a 93.5% confidence level is approximately the same as the VaR value at 99% confidence level.
        0.0008766274547215847

        References
        ----------
        [Risk.net](https://www.risk.net/definition/value-at-risk-var)
        [PELVE: Probability Equivalent Level of VaR and ES](https://www.sciencedirect.com/science/article/pii/S0304407622000380)

        """
        confidence_level = 1 - alpha

        var_value_es = np.percentile(pnl, 100 - (confidence_level * 100), interpolation="lower")
        es_value = np.mean(pnl[pnl < var_value_es])

        loss = abs(es_value - var_value)

        return loss

    return objective
