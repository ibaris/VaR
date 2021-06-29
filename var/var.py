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
from scipy.stats import gumbel_r
from scipy.stats import norm
from .methods import (historic, parametric, monte_carlo, garch)

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
    daily_pnl : array
        An array with the total daily mean values.
    info : dict
        A dict with general information about the parsed data:
            * Daily Mean Return (float): Total daily mean return. The mean of the variable `daily_pnl`.
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
        self.__len_alpha = len(self.alpha)

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
        self.daily_pnl = pd.DataFrame(np.average(self.daily_return, 1, self.weights), index=self.daily_return.index,
                                      columns=["Daily PnL"])

        cov_matrix = self.daily_return.cov()

        self.info = {"Daily Mean Return": np.mean(self.daily_pnl.values),
                     "Daily Volatility": np.std(self.daily_pnl.values),
                     "Portfolio STD": np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))}

    def __repr__(self):
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(self.info["Daily Mean Return"] * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(self.info["Daily Volatility"], 2),
                                                              port_sigma_val=round(self.info["Portfolio STD"], 2))

        return head

    def historic(self, daily_pnl=None):
        """
        The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
        It then assumes that history will repeat itself, from a risk perspective.

        Parameters
        ----------
        daily_pnl : DataFrame, None
            A DataFrame with
        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia](https://www.investopedia.com/articles/04/092904.asp)

        """
        data = historic(self.daily_pnl, self.alpha)
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
        data = parametric(self.daily_pnl, self.alpha, self.info["Portfolio STD"])
        df = pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
        return df

    def monte_carlo(self, stressed=False):
        """
        The Monte Carlo Method involves developing a model for future stock price returns and running multiple
        hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
        generates trials, but by itself does not tell us anything about the underlying methodology.

        The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
        distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
        extreme value distribution, log-Weibull and Gompertz distributions.

        Parameters
        ---------
        stressed : bool
            Use the Stressed Monte Carlo Method. Default is False.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
        [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
        [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
        """
        data = monte_carlo(self.daily_pnl, self.alpha, stressed=stressed)
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
        data = garch(self.daily_pnl, self.alpha)
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
        method_parametric = self.parametric()
        method_historic = self.historic()
        method_monte_carlo = self.monte_carlo()
        method_stressed_monte_carlo = self.monte_carlo(stressed=True)
        method_garch = self.garch()
        summary = summary.append(method_parametric).append(method_historic).append(method_monte_carlo).append(
            method_stressed_monte_carlo).append(
            method_garch)
        idx = ['Parametric', 'Historical', 'Monte Carlo', 'Stressed Monte Carlo', 'GARCH']
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary

    def backtest_data(self, method, window_days=250):
        """
        Generate the Backtest data.

        Parameters
        ----------
        method : str
            Define a VaR calculation method:
                * 'h' or 'historical': VaR calculated with the historical method,
                * 'p' or 'parametric': VaR calculated with the parametric method,
                * 'mc' or 'monte carlo': VaR calculated with the monte carlo method,
                * 'smv' or 'stressed monte carlo': VaR calculated with the stressed monte carlo method,
                * 'g' or 'garch': VaR calculated with the garch method.
        window_days : int
            Backtest horizon in days.

        Returns
        -------
        out : pd.DataFrame
            A DataFrame object with Daily PnL, VaR and VaR exception values.
        """
        if method == "historical" or method == "h":
            method_applied = historic
            kwargs = {"daily_pnl": None, "alpha": self.alpha}
        elif method == "parametric" or method == "p":
            method = "p"
            method_applied = parametric
            kwargs = {"daily_pnl": None, "alpha": self.alpha, "daily_std": None}
        elif method == "monte carlo" or method == "mc":
            method_applied = monte_carlo
            kwargs = {"daily_pnl": None, "alpha": self.alpha, "stressed": False}
        elif method == "stressed monte carlo" or method == "smc":
            method_applied = monte_carlo
            kwargs = {"daily_pnl": None, "alpha": self.alpha, "stressed": True}
        elif method == "garch" or method == "g":
            method_applied = garch
            kwargs = {"daily_pnl": None, "alpha": self.alpha}

        else:
            raise ValueError("Method {0} not understood. Available methods are 'h' ('historical'), 'p' ('parametric'), "
                             "'mc' ('monte carlo'), 'smv' ('stressed monte carlo') and 'g' ('garch').")

        daily_var_table = pd.DataFrame()
        for i in range(self.n - window_days):
            daily_return_sample = self.daily_return[i:i + window_days]

            kwargs["daily_pnl"] = pd.DataFrame(np.average(daily_return_sample, 1, self.weights),
                                               index=daily_return_sample.index,
                                               columns=["Daily PnL"])

            if method == "p":
                cov_matrix = daily_return_sample.cov()
                daily_std = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
                kwargs["daily_std"] = daily_std

            daily_var_df = pd.DataFrame(dict(zip(self.__header, method_applied(**kwargs))),
                                        index=[daily_return_sample.index.max()])

            daily_var_table = daily_var_table.append(daily_var_df)

        daily_var_table.index = daily_var_table.index + pd.DateOffset(1)  # Adjustment for matching VaR and actual PnL
        df = pd.merge_asof(self.daily_pnl, daily_var_table, right_index=True, left_index=True)

        header = self.__header[0:self.__len_alpha]

        df[header[0] + " exception"] = np.where((df[header[-1]] < df['Daily PnL']) &
                                                (df['Daily PnL'] < df[header[0]]),
                                                'True', 'False')

        df[header[-1] + " exception"] = np.where(df['Daily PnL'] < df[header[-1]], 'True', 'False')

        df = df.dropna()

        return df
