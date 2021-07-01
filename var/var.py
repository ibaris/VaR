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

import logging
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch.utility.exceptions import ConvergenceWarning
from tqdm import trange

from .methods import (historic, parametric, monte_carlo, garch, cdar)

__all__ = ["VaR"]

# ----------------------------------------------------------------------------------------------
# Environmental Settings
# ----------------------------------------------------------------------------------------------
# Filter `ConvergenceWarning` of `arch` module.
logging.captureWarnings(True)
warnings.filterwarnings('always', category=ConvergenceWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))
warnings.warn("This is a ConvergenceWarning", category=ConvergenceWarning)

# Plot settings
sns.set()
sns.set_color_codes("dark")
sns.set_style("whitegrid")

# Pandas DataFrame display settings.
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)


# ----------------------------------------------------------------------------------------------
# Value at Risk Class
# ----------------------------------------------------------------------------------------------
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
            * Daily Mean PnL (float): Total daily mean return. The mean of the variable `daily_pnl`.
            * Daily Volatility (float) : Total daily volatility.  The std of the variable `daily_mean`.
            * Portfolio Volatility: The std of the whole portfolio weighted by the parsed weights.

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
            A list confidence intervals (alpha values) for VaR. If None, the default values are [0.05, 0.025, 0.01].

        Notes
        -----
        Note, that the length of the weights must the same as the amount of columns of the `daily_return` parameter.

        Examples
        --------

        """
        self.alpha = np.array([0.05, 0.025, 0.01]) if alpha is None else np.atleast_1d(alpha).sort()
        self.__len_alpha = len(self.alpha)

        if len(self.alpha) > 3:
            raise AssertionError("The amount of alpha should be 3.")

        confidence = [1 - item for item in self.alpha]
        header_var = ["VaR(" + str(item * 100) + ")" for item in confidence]
        header_cvar = ["CVaR(" + str(item * 100) + ")" for item in confidence]
        header_var.extend(header_cvar)

        self.header_cdar = ["CDaR(" + str(item * 100) + ")" for item in confidence]

        self.__header = header_var
        self.header_var = header_var[0:self.__len_alpha]
        self.header_cvar = header_var[self.__len_alpha:]
        self.header_var_exception = [item + " exception" for item in self.header_var]
        self.header_cvar_exception = [item + " exception" for item in self.header_cvar]
        self.header_cdar_exception = [item + " exception" for item in self.header_cdar]

        self.daily_return = daily_return
        self.weights = weights
        self.n = self.daily_return.index.shape[0]
        self.__max_date = self.daily_return.index.max()
        self.daily_pnl = pd.DataFrame(np.average(self.daily_return, 1, self.weights), index=self.daily_return.index,
                                      columns=["Daily PnL"])

        cov_matrix = self.daily_return.cov()

        self.info = {"Daily Mean PnL": np.mean(self.daily_pnl.values),
                     "Daily Volatility": np.std(self.daily_pnl.values),
                     "Portfolio Volatility": np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))}

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self):
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(self.info["Daily Mean PnL"] * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(self.info["Daily Volatility"] * 100, 4),
                                                              port_sigma_val=round(self.info["Portfolio Volatility"] * 100, 4))

        return head

    def __str__(self):
        return self.summary().to_string()

    # ----------------------------------------------------------------------------------------------
    # Private Methods
    # ----------------------------------------------------------------------------------------------
    def __get_data_range(self, data, begin_date, end_date):
        if begin_date is None and end_date is not None:
            return data.loc[:end_date]
        elif begin_date is not None and end_date is None:
            return data.loc[begin_date:]
        elif begin_date is not None and end_date is not None:
            return data.loc[begin_date:end_date]

        return data

    # ----------------------------------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------------------------------
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
        data = parametric(self.daily_pnl, self.alpha, self.info["Portfolio Volatility"])
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

    def cdar(self, comp=True):
        """
        Calculate the Conditional Drawdown at Risk (CDaR) of a returns series.

        Parameters
        ----------
        comp : bool
            If True (Default) use the compounded cumulative returns.

        Returns
        -------
        out : list
            A list object with Value at Risk values at different confidence intervals.

        Notes
        -----
        CDaR of 0.20 at α=0.01 level means that for 1% worst cases of historical portfolio drawdowns, the average
        drawdown is 20%.

        """
        data = cdar(self.daily_pnl, self.alpha, comp=comp)
        df = -1 * pd.DataFrame(dict(zip(self.__header, data)), index=[self.__max_date])
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
        # cdar_comp = self.cdar(comp=True)
        summary = summary.append(method_parametric).append(method_historic).append(method_monte_carlo).append(
            method_stressed_monte_carlo).append(
            method_garch)
        idx = ['Parametric', 'Historical', 'Monte Carlo', 'Stressed Monte Carlo', 'GARCH']
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary

    def backtest(self, method, window_days=250):
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
                             "'mc' ('monte carlo'), 'smv' ('stressed monte carlo') and 'g' ('garch').".format(method))

        function_name = method_applied.__name__
        str_method = function_name.replace("_", " ").title()

        if "monte" in function_name:
            if kwargs["stressed"]:
                str_method += " (Stressed)"

        pbar_message = "Backtest Data: {method} Method".format(method=str_method)

        var_dict = dict()
        for i in trange(self.n - window_days, desc=pbar_message, leave=True):
            daily_return_sample = self.daily_return[i:i + window_days]

            kwargs["daily_pnl"] = pd.DataFrame(np.average(daily_return_sample, 1, self.weights),
                                               index=daily_return_sample.index,
                                               columns=["Daily PnL"])

            if method == "p":
                cov_matrix = daily_return_sample.cov()
                daily_std = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
                kwargs["daily_std"] = daily_std

            var_dict[daily_return_sample.index.max()] = method_applied(**kwargs)

        daily_var_table = pd.DataFrame.from_dict(var_dict).T.astype("float")
        daily_var_table.index.name = str_method
        daily_var_table.columns = self.__header

        daily_var_table.index = daily_var_table.index + pd.DateOffset(1)  # Adjustment for matching VaR and actual PnL

        df = pd.merge_asof(self.daily_pnl, daily_var_table, right_index=True, left_index=True)

        df = df.apply(pd.to_numeric)
        df[self.header_var_exception[0]] = np.where((df[self.header_var[-1]] < df['Daily PnL']) &
                                                    (df['Daily PnL'] < df[self.header_var[0]]),
                                                    'True', 'False')
        df[self.header_var_exception[-1]] = np.where(df['Daily PnL'] < df[self.header_var[-1]], 'True', 'False')

        df[self.header_cvar_exception[0]] = np.where((df[self.header_cvar[-1]] < df['Daily PnL']) &
                                                     (df['Daily PnL'] < df[self.header_cvar[0]]),
                                                     'True', 'False')

        df[self.header_cvar_exception[-1]] = np.where(df['Daily PnL'] < df[self.header_cvar[-1]], 'True', 'False')

        df = df.dropna()
        df.index.name = str_method

        return df

    def evaluate(self, backtest_data, begin_date=None, end_date=None):
        """
        Evaluate the backtest results.

        Parameters
        ----------
        backtest_data : pd.DataFrame
            The result of the function `backtest`.
        begin_date, end_date : str or None
            A begin and end date. If None, all data points will be considered.

        Returns
        -------
        out : pd.DataFrame
            A DataFrame object with following columns:
                * Amount : The Amount of the Observations or the VaR and CVaR exceptions.
                * Amount in Percent : The Amount of the Observations or the VaR and CVaR exceptions in percent.
                * Mean Deviation : The Mean Deviation of the exceptions (Actual - Expected).
                * STD Deviation : The Standard Deviation of the exceptions.
                * Min Deviation : The Min Deviation of the exceptions. This means the worst overestimation.
                * Max Deviation : The Max Deviation of the exceptions. This means the worst underestimation.

        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        observations = len(table)
        header = [self.header_var[0], self.header_var[-1]]
        header_cvar = [self.header_cvar[0], self.header_cvar[-1]]

        exception_var_data = [table[table[item] == 'True'] for item in [self.header_var_exception[0], self.header_var_exception[-1]]]
        exception_cvar_data = [table[table[item] == 'True'] for item in [self.header_cvar_exception[0], self.header_cvar_exception[-1]]]

        deviation_var = [item['Daily PnL'] - item[header[i]] for i, item in enumerate(exception_var_data)]
        deviation_cvar = [item['Daily PnL'] - item[header_cvar[i]] for i, item in enumerate(exception_cvar_data)]

        statistics_var = [(item.mean(), item.std(), item.max(), item.min()) for item in [np.concatenate([item.values for item in deviation_var]), deviation_var[-1]]]
        statistics_cvar = [(item.mean(), item.std(), item.max(), item.min()) for item in [np.concatenate([item.values for item in deviation_var]), deviation_cvar[-1]]]

        count_var = [np.sum([item.count() for item in deviation_var]), deviation_var[-1].count()]
        count_var_pct = [item / observations for item in count_var]

        count_cvar = [np.sum([item.count() for item in deviation_cvar]), deviation_cvar[-1].count()]
        count_cvar_pct = [item / observations for item in count_cvar]

        columns = ["Amount", "Amount in Percent", "Mean Deviation", "STD Deviation", "Min Deviation", "Max Deviation"]
        index = ["Observations"]
        index.extend(header)
        index.extend(header_cvar)

        df = pd.DataFrame(columns=columns, index=index)

        df.iloc[0] = [observations, 1, 0, 0, 0, 0]

        df.iloc[1] = [count_var[0], count_var_pct[0], statistics_var[0][0], statistics_var[0][1], statistics_var[0][2],
                      statistics_var[0][3]]
        df.iloc[2] = [count_var[1], count_var_pct[1], statistics_var[1][0], statistics_var[1][1], statistics_var[1][2],
                      statistics_var[1][3]]
        df.iloc[3] = [count_cvar[0], count_cvar_pct[0], statistics_cvar[0][0], statistics_cvar[0][1], statistics_cvar[0][2],
                      statistics_cvar[0][3]]
        df.iloc[4] = [count_cvar[1], count_cvar_pct[1], statistics_cvar[1][0], statistics_cvar[1][1], statistics_cvar[1][2],
                      statistics_cvar[1][3]]

        return df

    def var_plot(self, backtest_data, begin_date=None, end_date=None):
        """
        Plot the Value at Risk backtest data.

        Parameters
        ----------
        backtest_data : pd.DataFrame
            The result of the function `backtest`.
        begin_date, end_date : str or None
            A begin and end date. If None, all data points will be considered.

        Returns
        -------
        None
        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(table['Daily PnL'], color='#003049')
        ax.plot(table[self.header_var[0]], ":", color='#FF7600', alpha=0.7)
        ax.plot(table[self.header_var[-1]], "-.", color='#9d0208', alpha=0.7)

        exceed_0 = table[table[self.header_var_exception[0]] == 'True']['Daily PnL']
        exceed_1 = table[table[self.header_var_exception[-1]] == 'True']['Daily PnL']

        ax.scatter(exceed_0.index, exceed_0, marker='s', facecolors='none', edgecolors='#FF7600', s=120)
        ax.scatter(exceed_1.index, exceed_1, marker='x', facecolors='#9d0208', s=120)

        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(['Daily PnL',
                   self.header_var[0],
                   self.header_var[-1],
                   self.header_var_exception[0],
                   self.header_var_exception[-1]],
                  loc='upper left', prop={'size': 12})

        ax.set_title(backtest_data.index.name + ' VaR Backtest', fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()

    def cvar_plot(self, backtest_data, begin_date=None, end_date=None):
        """
        Plot the Conditional Value at Risk backtest data.

        Parameters
        ----------
        backtest_data : pd.DataFrame
            The result of the function `backtest`.
        begin_date, end_date : str or None
            A begin and end date. If None, all data points will be considered.

        Returns
        -------
        None
        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        daily_loss = table[table["Daily PnL"] < 0]

        exceed_1 = daily_loss[daily_loss[self.header_cvar_exception[-1]] == 'True']['Daily PnL']

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(daily_loss.index, daily_loss['Daily PnL'], label='Actual Loss', alpha=1, color='#2940D3')
        ax.plot(daily_loss.index, daily_loss[self.header_cvar[-1]], label=self.header_cvar[-1], color='deeppink', alpha=0.7)
        ax.scatter(exceed_1.index, exceed_1, marker='s', facecolors='none', edgecolors='deeppink', s=120)

        ax.legend(loc=3, prop={'size': 12})
        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(table.index.name + ' CVaR Backtest', fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()
