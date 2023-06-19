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
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch.utility.exceptions import ConvergenceWarning
from tqdm import trange
from scipy.optimize import dual_annealing
from var.auxiliary import array_like
from var.methods import garch, historic, monte_carlo, parametric
from var import objectives

__all__ = ["VaR"]

__METHODS__ = {"h": historic, "p": parametric, "mc": monte_carlo, "g": garch}
__PELVE_OBJECTIVES__ = {"h": objectives.pelve_historic, "p": objectives.pelve_parameteric}

# ----------------------------------------------------------------------------------------------
# Environmental Settings
# ----------------------------------------------------------------------------------------------
# Filter `ConvergenceWarning` of `arch` module.
logging.captureWarnings(True)
warnings.filterwarnings('always', category=ConvergenceWarning, module=r'^{0}\.'.format(re.escape(__name__)))
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
#pylint: disable=too-many-instance-attributes
class VaR:
    """
    The class to estimate the Value at Risk (VaR), Conditional Value at Risk or Expected Shortfall (ES) and the Conditional Drawdown at risk. The
    VaR can be calculated using different techniques like:
        * Parametric Method
        * Historical Method
        * Monte Carlo Method
        * GARCH Method

    Attributes
    ----------
    alpha : array
        Displays the array where the confidence level is stored.
    returns : DataFrame
        The parsed DataFrame object with the daily returns.
    weights : array
        Display the parsed weights.
    n : int
        Length of the parameter `returns`.
    pnl : array
        An array with the total daily mean values.
    info : dict
        A dict with general information about the parsed data:
            * Daily Mean PnL (float): Total daily mean return. The mean of the variable `pnl`.
            * Daily Volatility (float) : Total daily volatility.  The std of the variable `daily_mean`.
            * Portfolio Volatility: The std of the whole portfolio weighted by the parsed weights.
    methods : list
        A list with available test methods.

    References
    ----------
    [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)
    """

    def __init__(self, returns: pd.DataFrame, weights: array_like, alpha: Union[array_like, None] = None, **kwargs):
        """
        Initialize the Value-at-Risk class instance.

        Parameters
        ----------
        returns : pd.DataFrame
            A DataFrame object where the columns are the asset daily returns where the index is the corresponding date.
        weights : array_like
            An array with different weights corresponding to the assets.
        alpha : Union[array_like, None]
            A list significance levels (alpha values) for VaR. If None, the default values are [0.05, 0.025, 0.01].

        Notes
        -----
        Note, that the length of the weights must the same as the amount of columns of the `returns` parameter.

        """
        # if distribution not in list(get_args(distributions)):
        #     raise ValueError(
        #         f"Distribution {distribution} not available. Available distributions are {list(get_args(distributions))}."
        #     )

        self.alpha = np.array([0.05, 0.025, 0.01]) if alpha is None else np.atleast_1d(alpha)
        self.alpha.sort()
        self.alpha = self.alpha[::-1]

        self.len_alpha = len(self.alpha)

        if self.len_alpha > 3:
            raise AssertionError("The amount of alpha should be 3.")

        confidence = 1 - self.alpha
        headers = ["VaR", "ES", "CDaR"]

        self.header = []
        for i in range(len(headers)):
            self.header.extend(["{0}(".format(headers[i]) + str(item * 100) + ")" for item in confidence])

        self.header_exception = [item + " exception" for item in self.header]

        self.returns = returns
        self.weights = weights
        self.n = self.returns.index.shape[0]
        self.__max_date = self.returns.index.max()
        self.pnl = pd.DataFrame(np.average(self.returns, 1, self.weights),
                                index=self.returns.index,
                                columns=["Daily PnL"])

        cov_matrix = self.returns.cov()

        self.info = {
            "Mean PnL": np.mean(self.pnl.values),
            "Volatility": np.std(self.pnl.values),
            "Portfolio Volatility": np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
        }

        self.methods = list(__METHODS__.keys())

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self):
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, " \
               "Portfolio {sigma}: {port_sigma_val}%>".format(mu=chr(956),
                                                              mu_val=round(self.info["Mean PnL"] * 100, 2),
                                                              sigma=chr(963),
                                                              sigma_val=round(self.info["Volatility"] * 100, 4),
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
            A DataFrame object with Value at Risk values at different significance levels.

        References
        ----------
        [investopedia](https://www.investopedia.com/articles/04/092904.asp)

        """
        data = historic(self.pnl.values, self.alpha)
        df = pd.DataFrame(dict(zip(self.header, data)), index=[self.__max_date])
        return df

    def parametric(self):
        """
        Under the parametric method, also known as variance-covariance method, VAR is calculated as a function of mean
        and variance of the returns series, assuming normal distribution.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different significance levels.

        References
        ----------
        [Risk.net](https://www.risk.net/definition/value-at-risk-var)
        """
        data = parametric(pnl=self.pnl.values, alpha=self.alpha, daily_std=self.info["Portfolio Volatility"])

        df = pd.DataFrame(dict(zip(self.header, data)), index=[self.__max_date])
        return df

    def monte_carlo(self):
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
            A DataFrame object with Value at Risk values at different significance levels.

        References
        ----------
        [investopedia 1](https://www.investopedia.com/articles/04/092904.asp)
        [investopedia 2](https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp)
        [SciPy Gumbel Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html)
        """

        # if self.__dist_name == "gumbel_r":
        #     pnl = self.pnl.resample('W').min().dropna().values
        # else:
        #     pnl = self.pnl.values

        data = monte_carlo(pnl=self.pnl.values, alpha=self.alpha)

        df = pd.DataFrame(dict(zip(self.header, data)), index=[self.__max_date])
        return df

    def garch(self):
        """
        This method estimates the Value at Risk with a generalised autoregressive conditional heteroskedasticity (GARCH)
        model.

        Returns
        -------
        out : DataFrame
            A DataFrame object with Value at Risk values at different significance levels.

        References
        ----------
        [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)

        """
        data = garch(self.pnl.values, self.alpha)
        df = pd.DataFrame(dict(zip(self.header, data)), index=[self.__max_date])
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
            A DataFrame object with Value at Risk values at different significance levels.

        See Also
        --------
        parametric
        historic
        monte_carlo
        garch
        """
        method_parametric = self.parametric()
        method_historic = self.historic()
        method_monte_carlo = self.monte_carlo()
        method_garch = self.garch()

        summary = pd.concat([method_parametric, method_historic, method_monte_carlo, method_garch], ignore_index=True)

        idx = ['Parametric', 'Historical', 'Monte Carlo', 'GARCH']
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary

    def backtest(self, method: str, window_days: int = 250):
        """
        Generate the Backtest data.

        Parameters
        ----------
        method : str
            Define a VaR calculation method:
                * 'h': VaR calculated with the historical method,
                * 'p': VaR calculated with the parametric method,
                * 'mc': VaR calculated with the monte carlo method,
                * 'g': VaR calculated with the garch method.
        window_days : int
            Backtest horizon in days.

        Returns
        -------
        out : pd.DataFrame
            A DataFrame object with Daily PnL, VaR and VaR exception values.
        """
        if method not in __METHODS__.keys():
            raise ValueError(
                f"Method {method} not understood. Available methods are 'h' ('historical'), 'p' ('parametric'), "
                "'mc' ('monte carlo'), 'smv' ('stressed monte carlo') and 'g' ('garch').")

        method_applied = __METHODS__[method]
        kwargs = {"pnl": None, "alpha": self.alpha}

        function_name = method_applied.__name__
        str_method = function_name.replace("_", " ").title()

        desc = f"Backtest: {str_method} Method"

        var_dict = {}
        for i in trange(self.n - window_days, desc=desc, leave=True):
            returns_sample = self.returns[i:i + window_days]

            pnl = np.average(returns_sample, 1, self.weights)

            # if method == "mc":
            #     if self.__dist_name == "gumbel_r":
            #         kwargs["pnl"] = pd.DataFrame(pnl, index=returns_sample.index,
            #                                      columns=["Daily PnL"]).resample('W').min().dropna().values

            #     else:
            #         kwargs["pnl"] = pnl

            if method == "p":
                kwargs["pnl"] = pnl
                cov_matrix = returns_sample.cov()
                daily_std = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
                kwargs["daily_std"] = daily_std
            else:
                kwargs["pnl"] = pnl

            var_dict[returns_sample.index.max()] = method_applied(**kwargs)

        daily_var_table = pd.DataFrame.from_dict(var_dict).T.astype("float")
        daily_var_table.index.name = str_method
        daily_var_table.columns = self.header

        daily_var_table.index = daily_var_table.index + pd.DateOffset(1)  # Adjustment for matching VaR and actual PnL

        df = pd.merge_asof(self.pnl, daily_var_table, right_index=True, left_index=True)
        df = df.apply(pd.to_numeric)

        df1 = df.filter(self.header)  #* This contains the VaR and ES values

        for i, _ in enumerate(self.header):
            df[self.header_exception[i]] = df['Daily PnL'] < df1.values[:, i]

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
                * Amount : The Amount of the Observations or the VaR and ES exceptions.
                * Amount in Percent : The Amount of the Observations or the VaR and ES exceptions in percent.
                * Mean Deviation : The Mean Deviation of the exceptions (Actual - Expected).
                * STD Deviation : The Standard Deviation of the exceptions.
                * Min Deviation : The Min Deviation of the exceptions. This means the worst overestimation.
                * Max Deviation : The Max Deviation of the exceptions. This means the worst underestimation.

        """
        # ----------------------------------------------------------------------------------------------
        # Environmental Variables
        # ----------------------------------------------------------------------------------------------
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        df1 = table.filter(self.header)  #* This contains the VaR and ES values
        df2 = table.filter(self.header_exception)  #* This contains the VaR and ES exceptions

        columns = ["Amount", "Percent", "Mean Deviation", "STD Deviation", "Min Deviation", "Max Deviation"]
        df = pd.DataFrame(columns=columns, index=self.header)

        # ----------------------------------------------------------------------------------------------
        # Compute Statistics
        # ----------------------------------------------------------------------------------------------
        # Percentages ======================================================================
        percentages = df2.mean().values  #* This contains the VaR and ES exceptions in percent
        amount = df2.sum().values  #* This contains the VaR and ES exceptions in amount

        # Statistics =======================================================================
        for i, _ in enumerate(self.header):
            var_val = df1.values[:, i][df2.values[:, i]]
            pnl = table['Daily PnL'][df2.values[:, i]]
            data = np.abs(pnl - var_val)

            mean_values = data.mean()
            min_values = data.min()
            max_values = data.max()
            std_values = data.std()

            df.iloc[i] = [amount[i], percentages[i], mean_values, std_values, min_values, max_values]

        return df

    def compute_pelve(self, method: str, alpha: float = 0.01) -> Tuple[float, float]:
        """
        PELVE is intended to help decide what confidence level to use when replacing Value at Risk (VaR) 
        with Expected Shortfall (ES) in risk assessments.
        PELVE is essentially a ratio or a multiplier. It tells us how to adjust the confidence 
        level when switching from VaR to ES so that we get an equivalent measure of risk. The formula 
        for calculating PELVE is ES_{1-cɛ}(X)=V aR_{1-ɛ}(X), where ε is a small number close to 0, X is 
        a loss random variable, and c is the PELVE.

        The idea here is to answer the question: if we replace VaR with ES in our risk models, how will
        that affect our estimated capital requirements? Will we need more capital to cover potential 
        losses, or less?

        Parameters
        ----------
        method : str
           Define a VaR calculation method:
                * 'h': VaR calculated with the historical method,
                * 'p': VaR calculated with the parametric method,
        alpha : float, optional
            Significance level, by default 0.01

        Returns
        -------
        Tuple[float, float]
            PELVE and the optimization error.

        Raises
        ------
        ValueError
            If method not present.
        
        Note
        ----
        I already tried to avoid a grid search like this but scipy optimize functions where not able to
        minimize the function.

        Thank you Osman Mahmud Kim for pointing the PELVE out.

        See Also
        --------
        [PELVE: Probability Equivalent Level of VaR and ES](https://www.sciencedirect.com/science/article/pii/S0304407622000380)

        """
        if method not in __METHODS__:
            raise ValueError(f"Method {method} not understood. Available methods are {list(__METHODS__.keys())}")

        if method not in __PELVE_OBJECTIVES__:
            raise ValueError(
                f"Method {method} not available for PELVE. Available methods are {list(__PELVE_OBJECTIVES__.keys())}")

        method_applied = __METHODS__[method]
        pelve_applied = __PELVE_OBJECTIVES__[method]

        alpha = np.array([alpha])

        kwargs = {'pnl': self.pnl.values.flatten(), 'alpha': alpha}

        if method == "p":
            kwargs.update({"daily_std": self.info["Portfolio Volatility"]})

        var_value = method_applied(**kwargs)[0]

        # Compute Objective Function =======================================================
        _ = kwargs.pop("alpha")  # Remove alpha from kwargs since it is not needed anymore.
        kwargs["var_value"] = var_value

        objective = pelve_applied(**kwargs)

        # Apply the Dual Annealing Algorithm ===============================================
        # Define the bounds for es_confidence_level.
        bounds = [(0, 1)]

        # Call dual_annealing to minimize the objective function.
        result = dual_annealing(objective, bounds, x0=[0.025])

        # Extract the optimal es_confidence_level from the result.
        optimal_es_confidence_level = result.x[0]

        # The PELVE is the ratio of the optimal ES confidence level to the VaR confidence level.
        pelve = optimal_es_confidence_level / alpha

        # Compute Error ====================================================================
        kwargs = {'pnl': self.pnl.values.flatten(), 'alpha': np.array([optimal_es_confidence_level])}

        if method == "p":
            kwargs.update({"daily_std": self.info["Portfolio Volatility"]})

        es_value = method_applied(**kwargs)[1]

        difference = np.abs(var_value - es_value)

        return pelve, difference

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

        header_list = []
        header_exception_list = []

        for windows in [0]:
            header_exception = self.header_exception[windows:windows + self.len_alpha]
            header = self.header[windows:windows + self.len_alpha]
            header_list.extend([header[0], header[-1]])
            header_exception_list.extend([header_exception[0], header_exception[-1]])

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(table['Daily PnL'], color='#003049')

        ax.plot(table[header_list[0]], ":", color='#FF7600', alpha=0.7)
        ax.plot(table[header_list[-1]], "-.", color='#9d0208', alpha=0.7)

        exceed_0 = table[table[header_exception_list[0]] == True]['Daily PnL']
        exceed_1 = table[table[header_exception_list[-1]] == True]['Daily PnL']

        ax.scatter(exceed_0.index, exceed_0, marker='s', facecolors='none', edgecolors='#FF7600', s=120)
        ax.scatter(exceed_1.index, exceed_1, marker='x', facecolors='#9d0208', s=120)

        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(['Daily PnL', header_list[0], header_list[-1], header_exception_list[0], header_exception_list[-1]],
                  loc='upper left',
                  prop={'size': 12})

        ax.set_title(backtest_data.index.name + ' VaR Backtest', fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()

    def es_plot(self, backtest_data, begin_date=None, end_date=None):
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

        header_list = []
        header_exception_list = []

        for windows in [3]:
            header_exception = self.header_exception[windows:windows + self.len_alpha]
            header = self.header[windows:windows + self.len_alpha]
            header_list.extend([header[0], header[-1]])
            header_exception_list.extend([header_exception[0], header_exception[-1]])

        daily_loss = table[table["Daily PnL"] < 0]

        exceed_1 = daily_loss[daily_loss[header_exception_list[0]] == True]['Daily PnL']
        exceed_2 = daily_loss[daily_loss[header_exception_list[-1]] == True]['Daily PnL']

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(daily_loss.index, daily_loss['Daily PnL'], label='Actual Loss', alpha=1, color='#2940D3')

        ax.plot(daily_loss.index, daily_loss[header[0]], label=header[0], color='#FF7600', alpha=0.7)
        ax.plot(daily_loss.index, daily_loss[header[-1]], label=header[-1], color='#9d0208', alpha=0.7)

        ax.scatter(exceed_1.index, exceed_1, marker='s', facecolors='none', edgecolors='deeppink', s=120)
        ax.scatter(exceed_2.index, exceed_2, marker='x', facecolors='#9d0208', s=120)

        ax.legend(loc=3, prop={'size': 12})
        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(table.index.name + ' ES Backtest', fontsize=16, fontweight=1)

        ax.legend(['Daily PnL', header_list[0], header_list[-1], header_exception_list[0], header_exception_list[-1]],
                  prop={'size': 12})

        plt.tight_layout()
        plt.show()

    def cdar_plot(self, backtest_data, begin_date=None, end_date=None):
        """
        Plot the Conditional Drawdown at Risk backtest data.

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

        header_list = []
        header_exception_list = []

        for windows in [6]:
            header_exception = self.header_exception[windows:windows + self.len_alpha]
            header = self.header[windows:windows + self.len_alpha]
            header_list.extend([header[0], header[-1]])
            header_exception_list.extend([header_exception[0], header_exception[-1]])

        daily_loss = table[table["Daily PnL"] < 0]

        exceed_1 = daily_loss[daily_loss[header_exception_list[0]] == True]['Daily PnL']
        exceed_2 = daily_loss[daily_loss[header_exception_list[-1]] == True]['Daily PnL']

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(daily_loss.index, daily_loss['Daily PnL'], label='Actual Loss', alpha=1, color='#2940D3')

        ax.plot(daily_loss.index, daily_loss[header[0]], label=header[0], color='#FF7600', alpha=0.7)
        ax.plot(daily_loss.index, daily_loss[header[-1]], label=header[-1], color='#9d0208', alpha=0.7)

        ax.scatter(exceed_1.index, exceed_1, marker='s', facecolors='none', edgecolors='deeppink', s=120)
        ax.scatter(exceed_2.index, exceed_2, marker='x', facecolors='#9d0208', s=120)

        ax.legend(loc=3, prop={'size': 12})
        ax.spines['bottom'].set_color('#b0abab')
        ax.spines['top'].set_color('#b0abab')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(table.index.name + ' CDaR Backtest', fontsize=16, fontweight=1)

        ax.legend(['Daily PnL', header_list[0], header_list[-1], header_exception_list[0], header_exception_list[-1]],
                  prop={'size': 12})

        plt.tight_layout()
        plt.show()
