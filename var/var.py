# -*- coding: utf-8 -*-
"""
Value at Risk
=============
*Created on 26.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

The search for appropriate risk measuring methodologies has been followed by increased financial uncertainty
worldwide. Financial turmoil and the increased volatility of financial markets have induced the design and
development of more sophisticated tools for measuring and forecasting risk. The most well known risk measure is
value at risk (VaR), which is defined as the maximum loss over a targeted horizon for a given level of confidence.
In other words, it is an estimation of the tails of the empirical distribution of financial losses. It can be used
in all types of financial risk measurement.

References
----------
[Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)
[investopedia](https://www.investopedia.com/articles/04/092904.asp)
[Wikipedia](https://en.wikipedia.org/wiki/Value_at_risk)
"""

import math
import time
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

__all__ = ["VaR"]


class VaR:
    """
    The class to estimate the Value at Risk (VaR). The VaR can be calculated using different techniques like:
        * Parametric Method
        * Historical Method
        * Monte Carlo Method
        * GARCH Method

    Attributes
    ----------
    alpha : array
        Displays the array where the confidence level is stored.
    daily_return : DataFrame
        The parsed DataFrame object with the daily returns.
    weights : array
        Display the parsed weights.
    n : int
        Length of the parameter `daily_return`.
    daily_mean : array
        An array with the total daily mean values.
    info : dict
        A dict with general information about the parsed data:
            * Daily Mean Return (float): Total daily mean return. The mean of the variable `daily_mean`.
            * Daily Volatility (float) : Total daily volatility.  The std of the variable `daily_mean`.
            * Portfolio STD: The std of the whole portfolio weighted by the parsed weights.

    References
    ----------
    [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)
    """

    def __init__(self, daily_return, weights, alpha=None):
        """
        Initialize the Value-at-Risk class instance.

        Parameters
        ----------
        daily_return : DataFrame
            A DataFrame object where the columns are the asset daily returns where the index is the corresponding date.
        weights : array
            An array with different weights corresponding to the assets.
        alpha : list or None
            A list confidence intervals (alpha values) for VaR. If None, the default values are [5%, 2.5%, 1%].

        Notes
        -----
        Note, that the length of the weights must the same as the amount of columns of the `daily_return` parameter.

        Examples
        --------

        """
        self.alpha = np.array([5, 2.5, 1]) if alpha is None else np.atleast_1d(alpha)

        if len(self.alpha) > 3:
            raise AssertionError("The amount of alpha should be 3.")

        confidence = [100 - item for item in self.alpha]
        header_var = ["VaR(" + str(item) + ")" for item in confidence]
        header_cvar = ["CVaR(" + str(item) + ")" for item in confidence]
        header_var.extend(header_cvar)

        self.__header = header_var

        self.daily_return = daily_return
        self.weights = weights
        self.n = self.daily_return.index.shape[0]
        self.__max_date = self.daily_return.index.max()
        self.daily_mean = np.average(self.daily_return, 1, self.weights)

        cov_matrix = self.daily_return.cov()

        self.info = {"Daily Mean Return": np.mean(self.daily_mean),
                     "Daily Volatility": np.std(self.daily_mean),
                     "Portfolio STD": np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))}

    def __repr__(self):
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(self.info["Daily Mean Return"] * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(self.info["Daily Volatility"], 2),
                                                              port_sigma_val=round(self.info["Portfolio STD"], 2))

        return head

    def historic(self):
        """
        The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
        It then assumes that history will repeat itself, from a risk perspective.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia](https://www.investopedia.com/articles/04/092904.asp)

        """
        var_values = np.percentile(self.daily_mean, self.alpha)
        cvar_values = [np.mean(self.daily_mean[self.daily_mean <= item]) for item in var_values]
        data = np.append(var_values, cvar_values).flatten()

        df = pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
        return df

    def parametric(self):
        """
        Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
        and variance of the returns series, assuming normal distribution.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [Risk.net](https://www.risk.net/definition/value-at-risk-var)
        """
        z_values = norm.ppf(self.alpha / 100)

        var_values = self.info["Daily Mean Return"] + z_values * self.info["Portfolio STD"]
        cvar_values = [np.mean(self.daily_mean[self.daily_mean <= item]) for item in var_values]
        data = np.append(var_values, cvar_values).flatten()

        df = pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
        return df

    def monte_carlo(self, iv=None):
        """
        The Monte Carlo Method involves developing a model for future stock price returns and running multiple
        hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
        generates trials, but by itself does not tell us anything about the underlying methodology.

        Parameters
        ---------
        iv : int, float or None
            Run the Simulation with another volatility value, such that a Implied Volatility, instead of the volatility
            value derived from the daily returns.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia](https://www.investopedia.com/articles/04/092904.asp)
        """
        sigma = self.info["Daily Volatility"] if iv is None else iv

        PnL_list = np.random.normal(self.info["Daily Mean Return"], sigma, 500000)

        var_values = np.percentile(PnL_list, self.alpha)
        PnL_frame = [PnL_list[PnL_list <= item] for item in var_values]
        cvar_values = [np.nanmean(item) for item in PnL_frame]
        data = np.append(var_values, cvar_values).flatten()

        df = pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
        return df

    def garch(self):
        """
        This method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
        model.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(self.daily_mean, vol='Garch', p=1, o=0, q=1, dist='skewt')
            res = am.fit(disp='off')
            forecasts = res.forecast(reindex=True)

        q = am.distribution.ppf(self.alpha / 100, res.params[-2:])
        value_at_risk = forecasts.mean.values + np.sqrt(forecasts.variance).values * q[None, :]

        var_values = [value_at_risk[[-1, ]][0][0], value_at_risk[[-1, ]][0][1], value_at_risk[[-1, ]][0][2]]
        cvar_values = [np.mean(self.daily_mean[self.daily_mean <= item]) for item in var_values]
        cvar_values = [var_values[-1] if math.isnan(item) else item for item in cvar_values]

        data = np.append(var_values, cvar_values).flatten()

        df = pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
        return df

    def summary(self):
        """
        Summary of Value-at-Risk with different models:
            * Parametric Method
            * Historical Method
            * Monte Carlo Method
            * GARCH Method

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        See Also
        --------
        parametric
        historic
        monte_carlo
        garch
        """
        summary = pd.DataFrame()
        parametric = self.parametric()
        historic = self.historic()
        monte_carlo = self.monte_carlo()
        garch = self.garch()
        summary = summary.append(parametric).append(historic).append(monte_carlo).append(garch)
        idx = ['Parametric', 'Historical', 'Monte Carlo', 'GARCH']
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary
