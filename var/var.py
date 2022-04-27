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

from var.methods import (__METHODS__, historic, parametric, monte_carlo, monte_carlo_stressed, cdar)
from var.auxiliary import array_like, number, data_frame

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

__XaR__ = {"var": 0,
           "cvar": 1,
           "cdar": 2}

__TITLE__ = ["VaR", "CVaR", "CDaR"]

# ----------------------------------------------------------------------------------------------
# Value at Risk Class
# ----------------------------------------------------------------------------------------------
class VaR:
    """
    The class to estimate the Value at Risk (VaR), Conditional Value at Risk (CVaR) and the Conditional Drawdown at risk. The
    VaR can be calculated using different techniques like:
        * Parametric Method
        * Historical Method
        * Monte Carlo Method

    Attributes
    ----------
    alpha : number
        Displays the array where the confidence level is stored.
    daily_return : data_frame
        The parsed DataFrame object with the daily returns.
    weights : array_like
        Display the parsed weights.
    n : int
        Length of the parameter `daily_return`.
    pnl : array_like
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

    def __init__(self, daily_return, weights: array_like = None, alpha: number = 0.01) -> None:
        """
        Initialize the Value-at-Risk class instance.

        Parameters
        ----------
        daily_return : data_frame
            A DataFrame object where the columns are the asset daily returns where the index is the corresponding date.
        weights : array_like, optional
            An array with different weights corresponding to the assets. Default is 1.
        alpha : number, optional
            A confidence interval (alpha) for VaR, by default 0.01.

        Notes
        -----
        Note, that the length of the weights must the same as the amount of columns of the `daily_return` parameter.

        """
        self.alpha = alpha

        confidence = 1 - self.alpha
        headers = ["VaR", "CVaR", "CDaR"]

        self.header = [f"{item}({confidence})" for item in headers]
        self.header_exception = [item + " exception" for item in self.header]

        self.daily_return = daily_return
        self.weights = np.array([1]) if weights is None else np.atleast_1d(weights)
        self.n = self.daily_return.index.shape[0]
        self.__max_date = self.daily_return.index.max()
        self.pnl = pd.DataFrame(np.average(self.daily_return, 1, self.weights), index=self.daily_return.index,
                                      columns=["Daily PnL"])

        cov_matrix = self.daily_return.cov()

        self.info = {"Daily Mean PnL": np.mean(self.pnl.values),
                     "Daily Volatility": np.std(self.pnl.values),
                     "Portfolio Volatility": np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))}

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self):
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(
                                                                  self.info["Daily Mean PnL"] * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(
                                                                  self.info["Daily Volatility"] * 100, 4),
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
    def historic(self) -> data_frame:
        """Historic

        The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
        It then assumes that history will repeat itself, from a risk perspective.

        Returns
        -------
        out : data_frame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia](https://www.investopedia.com/articles/04/092904.asp)

        """
        data = historic(self.pnl.values, self.alpha)

        df = pd.DataFrame(dict(zip(self.header[:-1], data)), index=[self.__max_date])

        return df

    def parametric(self) -> data_frame:
        """Parametric

        Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
        and variance of the returns series, assuming normal distribution.

        Returns
        -------
        out : data_frame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [Risk.net](https://www.risk.net/definition/value-at-risk-var)
        """
        data = parametric(self.pnl.values, self.alpha, self.info["Portfolio Volatility"])

        df = pd.DataFrame(dict(zip(self.header[:-1], data)), index=[self.__max_date])

        return df

    def monte_carlo(self, stressed=False) -> data_frame:
        """Monte Carlo

        The Monte Carlo Method involves developing a model for future stock price returns and running multiple
        hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
        generates trials, but by itself does not tell us anything about the underlying methodology.

        The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
        distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
        extreme value distribution, log-Weibull and Gompertz distributions.

        Parameters
        ---------
        stressed : bool, optional
            Use the Stressed Monte Carlo Method. Default is False.

        Returns
        -------
        out : data_frame
            A DataFrame object with Value at Risk values at different confidence intervals.

        References
        ----------
        [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
        [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
        [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
        """
        if stressed:
            block_maxima = self.pnl.resample('W').min().dropna().values
            data = monte_carlo_stressed(block_maxima, self.alpha)
        else:
            data = monte_carlo(self.pnl.values, self.alpha)

        df = pd.DataFrame(dict(zip(self.header[:-1], data)), index=[self.__max_date])

        return df

    def cdar(self) -> data_frame:
        """CDaR

        Calculate the Conditional Drawdown at Risk (CDaR) of a returns series.

        Returns
        -------
        data_frame
            Conditional Drawdown at Risk at desired confidence interval.
        """
        CDaR = cdar(self.pnl.values, self.alpha)

        df = pd.DataFrame(columns=[self.header[-1]],
                          index=[self.__max_date])

        df.iloc[0] = CDaR
        
        return df

    def summary(self) -> data_frame:
        """Summary

        Summary of Value-at-Risk with different models:
            * Parametric Method
            * Historical Method
            * Monte Carlo Method

        Returns
        -------
        out : data_frame
            A DataFrame object with Value at Risk values at different confidence intervals.

        See Also
        --------
        parametric
        historic
        monte_carlo
        """
        methods = [self.parametric(), self.historic(), self.monte_carlo(), self.monte_carlo(stressed=True)]

        CDaR = cdar(self.pnl.values, self.alpha)

        summary = pd.DataFrame()
        for method in methods:
            summary = summary.append(method)

        summary[self.header[-1]] = CDaR

        idx = ['Parametric', 'Historical', 'Monte Carlo', 'Stressed Monte Carlo']
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary

    def backtest(self, method: str, window_days: int = 250) -> data_frame:
        """Backtest

        Generate the Backtest data.

        Parameters
        ----------
        method : str
            Define a VaR calculation method:
                * 'h': VaR calculated with the historical method,
                * 'p': VaR calculated with the parametric method,
                * 'mc': VaR calculated with the monte carlo method,
                * 'smc': VaR calculated with the stressed monte carlo method,
        window_days : int, optional
            Backtest horizon in days, by default 250.

        Returns
        -------
        out : data_frame
            A DataFrame object with Daily PnL, VaR and VaR exception values.
        """
        method_applied = __METHODS__[method]
        kwargs = {"pnl": None, "alpha": self.alpha}

        if method not in __METHODS__.keys():
            raise ValueError("Method {0} not understood. Available methods are 'h' ('historical'), 'p' ('parametric'), "
                             "'mc' ('monte carlo') and 'smc' ('stressed monte carlo').".format(method))

        function_name = method_applied.__name__
        str_method = function_name.replace("_", " ").title()

        desc = "Backtest: {method} Method".format(method=str_method)

        var_dict = dict()
        for i in trange(self.n - window_days, desc=desc, leave=True):
            daily_return_sample = self.daily_return[i:i + window_days]

            daily_pnl = np.average(daily_return_sample, 1, self.weights)

            if method == "smc":
                kwargs["pnl"] = pd.DataFrame(daily_pnl, index=daily_return_sample.index,
                                             columns=["Daily PnL"]).resample('W').min().dropna().values

            elif method == "p":
                kwargs["pnl"] = daily_pnl
                cov_matrix = daily_return_sample.cov()
                daily_std = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
                kwargs["daily_std"] = daily_std
            else:
                kwargs["pnl"] = daily_pnl

            data = method_applied(**kwargs)
            CDaR = cdar(kwargs["pnl"], self.alpha)

            var_dict[daily_return_sample.index.max()] = [data[0], data[1], CDaR]

        daily_var_table = pd.DataFrame.from_dict(var_dict, orient="index").astype("float")
        daily_var_table.index.name = str_method
        daily_var_table.columns = self.header

        daily_var_table.index = (daily_var_table.index +
                                 pd.DateOffset(1))  # Adjustment for matching VaR and actual PnL

        df = pd.merge_asof(self.pnl, daily_var_table, right_index=True, left_index=True)

        df = df.apply(pd.to_numeric)

        df[self.header_exception[0]] = np.where(df['Daily PnL'] < df[self.header[0]],
                                                'True', 'False')

        df[self.header_exception[1]] = np.where(df['Daily PnL'] < df[self.header[1]],
                                                'True', 'False')

        df[self.header_exception[2]] = np.where(df['Daily PnL'] < df[self.header[2]],
                                                'True', 'False')

        df = df.dropna()
        df.index.name = str_method

        return df

    def evaluate(self, backtest_data: data_frame, begin_date: str = None, end_date: str = None) -> data_frame:
        """Evaluate

        Evaluate the backtest results.

        Parameters
        ----------
        backtest_data : data_frame
            The result of the function `backtest`.
        begin_date, end_date : str or None, optional
            A begin and end date. If None (default), all data points will be considered.

        Returns
        -------
        out : data_frame
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

        tmp_exc = [table[table[item] == 'True'] for item in self.header_exception]
        tmp_dev = [item['Daily PnL'] - item[self.header[i]] for i, item in enumerate(tmp_exc)]

        try:
            tmp_stat = list()

            for item in tmp_dev:
                mean = item.mean()
                std = item.std()
                max_i = item.max()
                min_i = item.min()

                if len(item) == 1:
                    std = 0
                    max_i = 0
                    min_i = 0

                tmp_stat.append((mean, std, max_i, min_i))

        except (NameError, ValueError):
            tmp_stat = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]

        count = [item.count() for item in tmp_dev]
        count_pct = [item / observations for item in count]

        columns = ["Amount", "Percent", "Mean Deviation", "STD Deviation", "Min Deviation", "Max Deviation"]
        df = pd.DataFrame(columns=columns, index=self.header)

        for i, idx in enumerate(self.header):
            item_list = [count[i], count_pct[i]]
            item_list.extend(tmp_stat[i])

            df.loc[idx] = item_list

        return df

    def plot(self, backtest_data: data_frame, method: str = "var", begin_date: str = None,
             end_date: str = None, figsize=(14, 4)):
        """
        Plot the Value at Risk backtest data.

        Parameters
        ----------
        backtest_data : data_frame
            The result of the function `backtest`.
        method : str, optional
            Determine the quantity to be plotted:
                * var (default): Value at Risk,
                * cvar : Conditional Value at Risk,
                * cdar : Conditional Drawdown at Risk.
        begin_date, end_date : str or None, optional
            A begin and end date. If None, all data points will be considered.
        figsize : Tuple(int, int), optional
            The subplot figure size, by default (14, 4).

        Returns
        -------
        fig, ax
            Returns the figure and axis instances of the matplotlib subplot.
        """
        if method not in list(__XaR__.keys()):
            raise AssertionError("Method should be `var`, `cvar` or `cdar`.")

        c = __XaR__[method]

        table = self.__get_data_range(backtest_data, begin_date, end_date)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(table['Daily PnL'], color='#003049')

        ax.plot(table[self.header[c]], "-.", color='#9d0208', alpha=0.7)

        exceed_0 = table[table[self.header_exception[c]] == 'True']['Daily PnL']

        ax.scatter(exceed_0.index, exceed_0, marker='x', facecolors='#9d0208', s=120)

        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(['Daily PnL',
                   self.header[c],
                   "Exceptions"],
                  loc='upper left', prop={'size': 12})

        ax.set_title(backtest_data.index.name + f' {__TITLE__[c]} Backtest', fontsize=16, fontweight=1)

        fig.tight_layout()
        plt.show()

        return fig, ax
