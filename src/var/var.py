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

from __future__ import annotations

import itertools
import logging
import re
import time
import warnings
from typing import Literal, Optional, Sequence, Tuple, Union, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch.utility.exceptions import ConvergenceWarning
from fitter import Fitter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import dual_annealing
from tqdm import trange

from var import objectives
from var.auxiliary import __DISTRIBUTIONS__, array_like, distributions
from var.methods import garch, historic, monte_carlo, parametric

__all__ = ["VaR"]

__METHODS__ = {"h": historic, "p": parametric, "mc": monte_carlo, "g": garch}
__PELVE_OBJECTIVES__ = {"h": objectives.pelve_historic, "p": objectives.pelve_parameteric}

# ----------------------------------------------------------------------------------------------
# Environmental Settings
# ----------------------------------------------------------------------------------------------
# Filter `ConvergenceWarning` of `arch` module.
logging.captureWarnings(True)
warnings.filterwarnings("always", category=ConvergenceWarning, module=r"^{0}\.".format(re.escape(__name__)))
warnings.warn("This is a ConvergenceWarning", stacklevel=2, category=ConvergenceWarning)

# Plot settings
sns.set()
sns.set_color_codes("dark")
sns.set_style("whitegrid")

# Pandas DataFrame display settings.
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 15)

# List of colors
colors = ["#FF7600", "#9d0208", "#9d0255", "c", "m", "y", "k"]
# List of line styles
line_styles = ["-", "--", "-.", ":"]
# List of marker
markers = ["s", "d", "o", "v", "x"]

# Create cycle iterators
color_cycle = itertools.cycle(colors)
line_style_cycle = itertools.cycle(line_styles)
marker_cycle = itertools.cycle(markers)


# ----------------------------------------------------------------------------------------------
# Value at Risk Class
# ----------------------------------------------------------------------------------------------
# pylint: disable=too-many-instance-attributes
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
    header : list
        A list with the header of the DataFrame object.
    header_exception : list
        A list with the header of the DataFrame object for the exception.
    _portfolio_volatility : float
        The portfolio volatility.
    _mean_pnl : float
        The daily mean.
    _volatility : float
        The daily volatility.

    References
    ----------
    [Julija Cerović Smolović, 2017](https://doi.org/10.1080/1331677X.2017.1305773)
    [investopedia](https://www.investopedia.com/articles/04/092904.asp)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: array_like | None = None,
        alpha: array_like | None = None,
        freq: Literal["B", "D", "W", "M", "Q", "Y", "h", "min", "s", "ms", "us", "ns"] = "D",
        distribution: Literal["chi2", "gamma", "lognorm", "norm", "uniform", "t", "gumbel_r", "f"] = "norm",
        **kwargs: dict,
    ) -> None:
        """
        Initialize the Value-at-Risk class instance.

        Parameters
        ----------
        returns : pd.DataFrame
            A DataFrame object where the columns are the asset daily returns where the index is the corresponding date.
        weights : Sequence, optional
            The weights corresponding to the assets in the portfolio. If None, then the weights will be equal to 1/n,
            where n is the number of assets in the portfolio. Default is None. If the sum of the weights is not equal to 1,
            then the weights will be normalized to sum to 1.
        alpha : Union[array_like, None]
            A list significance levels (alpha values) for VaR. If None, the default values are [0.05, 0.025, 0.01].
        distribution : Literal["chauchy", "chi2", "expon", "exponpow", "gamma", "lognorm", "norm", "powerlaw", "rayleigh", "uniform", "t", "gumbel_r", "f"]
            The distribution to use for the VaR calculation. The default is "norm". If the distribution is
            `norm` or `t`, the mean and the standard deviation of the returns are used. The degree of
            freedom of the `t` distribution is set to `len(returns) - 1`.
        **kwargs : dict
            Additional keyword arguments for the distribution methods. The distribution kwargs are from the
            individual distribution methods of `scipy.stats`.

        Notes
        -----
        Note, that the length of the weights must the same as the amount of columns of the `returns` parameter.

        """
        self.freq = freq

        if distribution not in list(get_args(distributions)):
            raise ValueError(f"Distribution {distribution} not available. Available distributions are {list(get_args(distributions))}.")

        # Create Weights ===================================================================
        if weights is None:
            weights = np.ones(returns.shape[1]) / returns.shape[1]

        if len(weights) != returns.shape[1]:
            raise ValueError(
                "The length of the weights must be equal to the number of assets in the portfolio. "
                f"The current length of the weights is {len(weights)} and the number of assets in the portfolio is {returns.shape[1]}."
            )

        if np.sum(weights) != 1:
            weights = np.array(weights) / np.sum(weights)

        self.weights = weights

        # Create Alpha =====================================================================
        self.alpha = np.array([0.05, 0.025, 0.01]) if alpha is None else np.atleast_1d(alpha)
        self.alpha.sort()
        self.alpha = self.alpha[::-1]

        self.len_alpha = len(self.alpha)

        if self.len_alpha > 3:
            raise AssertionError("The maximum amount of alpha should be 3.")

        confidence = 1 - self.alpha

        # Create Header ====================================================================
        headers = ["VaR", "ES", "DD"]

        self.header = []
        for i in range(len(headers)):
            self.header.extend([f"{headers[i]}({item * 100!s})" for item in confidence])

        self.header_exception = [item + " exception" for item in self.header]

        # Compute General Information ======================================================
        self.returns = returns
        self.n = self.returns.shape[0]
        self.k = self.returns.shape[-1]
        self.__max_date = self.returns.index.max()
        self._pnl_header = f"PnL({self.freq})"

        self.pnl = pd.DataFrame(np.average(self.returns, 1, self.weights), index=self.returns.index, columns=[self._pnl_header])

        cov_matrix = self.returns.cov()

        self._portfolio_volatility = np.sqrt(self.weights.T.dot(cov_matrix).dot(self.weights))
        self._mean_pnl = np.mean(self.pnl.values)
        self._volatility = np.std(self.pnl.values)

        self.info = {"Mean PnL": self._mean_pnl, "Volatility": self._volatility, "Portfolio Volatility": self._portfolio_volatility}

        self.methods = list(__METHODS__.keys())

        # ----------------------------------------------------------------------------------------------
        # Distributions
        # ----------------------------------------------------------------------------------------------
        self.__dist_name = distribution
        self.distribution = __DISTRIBUTIONS__[distribution]
        self.kwargs = kwargs

        if distribution == "norm":
            self.kwargs["loc"] = self._mean_pnl
            self.kwargs["scale"] = self._volatility

        if distribution == "t":
            self.kwargs["df"] = len(self.returns) - 1
            self.kwargs["loc"] = self._mean_pnl
            self.kwargs["scale"] = self._volatility

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Returns a string representation of the VaR object, including mean PnL, volatility, and portfolio volatility.

        Returns
        -------
        str
            A formatted string displaying the mean PnL (μ), volatility (σ), and portfolio volatility (Portfolio σ) as percentages.
        """
        head = "<VaR - {mu}: {mu_val}%, {sigma}: {sigma_val}%, Portfolio {sigma}: {port_sigma_val}%>".format(
            mu=chr(956),
            mu_val=round(self._mean_pnl * 100, 2),
            sigma=chr(963),
            sigma_val=round(self._volatility * 100, 4),
            port_sigma_val=round(self._portfolio_volatility * 100, 4),
        )

        return head

    def __str__(self) -> str:
        return self.summary().to_string()

    # ----------------------------------------------------------------------------------------------
    # Private Methods
    # ----------------------------------------------------------------------------------------------
    def __get_data_range(
        self,
        data: pd.DataFrame,
        begin_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        """
        Return a slice of the DataFrame between begin_date and end_date.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to slice.
        begin_date : str or pd.Timestamp or None
            The start date for the slice. If None, slice from the beginning.
        end_date : str or pd.Timestamp or None
            The end date for the slice. If None, slice to the end.

        Returns
        -------
        pd.DataFrame
            The sliced DataFrame.
        """
        if begin_date is None and end_date is not None:
            return data.loc[:end_date]

        if begin_date is not None and end_date is None:
            return data.loc[begin_date:]

        if begin_date is not None and end_date is not None:
            return data.loc[begin_date:end_date]

        return data

    # ----------------------------------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------------------------------
    def fit_distributions(
        self,
        distribution: distributions | None = None,
        plot: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Fit a distribution to the returns data.

        Parameters
        ----------
        distribution: distributions or None, optional
            Choose a distribution to fit. If None (default)  consider the one you specified in the
            initialization process (default).
        plot : bool, optional
            Plot the best fitted distribution, by default False.
        verbose : bool, optional
            Print intermediate steps, by default False.
        """
        if distribution is None:
            distribution = get_args(distributions)

        f = Fitter(self.pnl.to_numpy().flatten(), distributions=distribution)
        f.fit(progress=verbose)

        if verbose:
            print("\n")
            print("Best fits:")
            print("----------")

        if plot or verbose:
            print(f.summary(plot=plot))

        best_fit = f.get_best(method="sumsquare_error")
        self.__dist_name = next(iter(best_fit.keys()))
        self.distribution = __DISTRIBUTIONS__[self.__dist_name]
        self.kwargs = best_fit[self.__dist_name]

        if verbose:
            print("\n")
            print("Best fit:")
            print("---------")
            print(f"Distribution {self.__dist_name} with parameters {self.kwargs}")

    @property
    def get_distribution(self) -> dict:
        """Return the Distribution parameter"""
        return {"name": self.__dist_name, "kwargs": self.kwargs}

    def historic(self) -> pd.DataFrame:
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
        data = historic(self.pnl.to_numpy().T, self.alpha, axis=1)  # PnL Shape: (1, T)
        return data.to_df([self.__max_date])

    def parametric(self) -> pd.DataFrame:
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
        kwargs = self.kwargs.copy()

        # kwargs.pop("loc", None)
        # kwargs.pop("scale", None)

        data = parametric(
            pnl=self.pnl.to_numpy().T,
            alpha=self.alpha,
            std=self._portfolio_volatility,
            ppf=self.distribution.ppf,
            axis=1,
            **kwargs,
        )

        return data.to_df([self.__max_date])

    def monte_carlo(self) -> pd.DataFrame:
        """
        The Monte Carlo Method involves developing a model for future stock price returns and running multiple
        hypothetical trials through the model. A Monte Carlo simulation refers to any method that randomly
        generates trials, but by itself does not tell us anything about the underlying methodology.

        The Stressed Monte Carlo Method uses the Gumel distribution to generate the random trials. The Gumbel
        distribution is sometimes referred to as a type I Fisher-Tippett distribution. It is also related to the
        extreme value distribution, log-Weibull and Gompertz distributions.

        Parameters
        ----------
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
        data = monte_carlo(
            pnl=self.pnl.to_numpy().T,
            alpha=self.alpha,
            rvs=self.distribution.rvs,
            **self.kwargs,
        )

        return data.to_df([self.__max_date])

    def garch(
        self,
        dist: Literal["normal", "gaussian", "t", "studentst", "ged", "generalized error"] = "normal",
    ) -> pd.DataFrame:
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
        data = garch(self.pnl.to_numpy().T, self.alpha, dist=dist)
        return data.to_df([self.__max_date])

    def summary(self) -> pd.DataFrame:
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

        idx = ["Parametric", "Historical", "Monte Carlo", "GARCH"]
        summary.index = idx
        summary.index.name = time.strftime("%Y-%m-%d")
        return summary

    def backtest(
        self,
        method: Literal["h", "p", "mc", "g"],
        window: int = 250,
        garch_dist: Literal["normal", "gaussian", "t", "studentst", "ged", "generalized error"] = "normal",
    ) -> pd.DataFrame:
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
        window : int
            Backtest horizon in the same unit as the returns. Default is 250.

        Returns
        -------
        out : pd.DataFrame
            A DataFrame object with Daily PnL, VaR and VaR exception values.

        Note
        ----
        The GARCH method will take a while, since it is not possible to vectorize it.
        """
        # Utilities =======================================================================
        if method not in __METHODS__:
            raise ValueError(
                f"Method {method} not understood. Available methods are 'h' ('historical'), 'p' ('parametric'), "
                "'mc' ('monte carlo'), 'smv' ('stressed monte carlo') and 'g' ('garch').",
            )

        method_applied = __METHODS__[method]
        kwargs = {"pnl": None, "alpha": self.alpha}

        # Add the distribution parameter to the keyword arguments.
        kwargs.update(self.kwargs)

        function_name = method_applied.__name__
        str_method = function_name.replace("_", " ").title()

        # Windowed Return Computation =====================================================
        pnl = self.returns @ self.weights

        pnl_w = sliding_window_view(pnl, window_shape=(window))[:-1]
        date_w = sliding_window_view(pnl.index, window_shape=(window))[:-1].max(axis=1) + +np.timedelta64(1, self.freq)

        kwargs["pnl"] = pnl_w
        kwargs["axis"] = 1

        # Methods --------------------------------------------------------------
        if method == "p":
            # Parametric -------------------------------------------
            std = pnl_w.std(axis=1)

            kwargs.pop("scale", None)
            kwargs.pop("loc", None)

            kwargs["std"] = std
            kwargs["ppf"] = self.distribution.ppf

        elif method == "mc":
            # Monte Carlo ----------------------------------------------------------
            kwargs["rvs"] = self.distribution.rvs

        elif method == "g":
            kwargs.pop("axis", None)
            kwargs.pop("scale", None)
            kwargs.pop("loc", None)

            kwargs.update({"dist": garch_dist})

        # Run Simulation -------------------------------------------------------
        sim = method_applied(**kwargs)

        # Post-Processing =================================================================
        sim = sim.to_df(index=date_w)

        df = pd.merge_asof(self.pnl, sim, right_index=True, left_index=True)
        df = df.apply(pd.to_numeric)

        masks = [df[self._pnl_header] < df[header] for header in self.header]

        acc = np.where(masks, True, False)

        df[self.header_exception] = acc.T

        df = df.dropna()
        df.name = str_method

        return df

    def evaluate(
        self,
        backtest_data: pd.DataFrame,
        begin_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
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

        # * This contains the VaR and ES values
        df1 = table.filter(self.header)
        # * This contains the VaR and ES exceptions
        df2 = table.filter(self.header_exception)

        columns = ["Total", "Exceptions", "Percent", "Mean Deviation", "STD Deviation", "Min Deviation", "Max Deviation"]
        df = pd.DataFrame(columns=columns, index=self.header)

        # ----------------------------------------------------------------------------------------------
        # Compute Statistics
        # ----------------------------------------------------------------------------------------------
        # Percentages ======================================================================
        # * This contains the VaR and ES exceptions in percent
        percentages = df2.mean().to_numpy()
        amount = df2.sum().to_numpy()  # * This contains the VaR and ES exceptions in amount

        # Statistics =======================================================================
        for i, _ in enumerate(self.header):
            var_val = df1.to_numpy()[:, i][df2.to_numpy()[:, i]]
            pnl = table[self._pnl_header][df2.to_numpy()[:, i]]
            data = np.abs(pnl - var_val)

            mean_values = data.mean()
            min_values = data.min()
            max_values = data.max()
            std_values = data.std()

            df.iloc[i] = [backtest_data.shape[0], amount[i], percentages[i], mean_values, std_values, min_values, max_values]

        return df

    def compute_pelve(self, method: Literal["h", "p"], alpha: float = 0.01) -> tuple[float, float]:
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
            raise ValueError(f"Method {method} not available for PELVE. Available methods are {list(__PELVE_OBJECTIVES__.keys())}")

        method_applied = __METHODS__[method]
        pelve_applied = __PELVE_OBJECTIVES__[method]

        alpha = np.array([alpha])

        kwargs = {"pnl": self.pnl.to_numpy().T, "alpha": alpha}

        if method == "p":
            kwargs.update({"daily_std": self._portfolio_volatility})

        var_value = method_applied(**kwargs).var[0, 0]

        # Compute Objective Function =======================================================
        # Remove alpha from kwargs since it is not needed anymore.
        _ = kwargs.pop("alpha")
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
        kwargs = {"pnl": self.pnl.to_numpy().T, "alpha": np.array([optimal_es_confidence_level])}

        if method == "p":
            kwargs.update({"daily_std": self._portfolio_volatility})

        es_value = method_applied(**kwargs).es[0, 0]

        difference = np.abs(var_value - es_value)

        return pelve, difference

    def var_plot(
        self,
        backtest_data: pd.DataFrame,
        begin_date: str | None = None,
        end_date: str | None = None,
        alpha_index: int | slice | None = None,
    ) -> tuple[Figure, Axes]:
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
        fig, axes : Figure, Axes
            Matplotlib figure and axes object.
        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        header_list = []
        header_exception_list = []

        for windows in [0]:
            header_exception = self.header_exception[windows : windows + self.len_alpha]
            header = self.header[windows : windows + self.len_alpha]
            header_list.extend(header)
            header_exception_list.extend(header_exception)

        if alpha_index is not None:
            header_list = header_list[alpha_index]
            header_exception_list = header_exception_list[alpha_index]

            if isinstance(header_list, str):
                header_list = [header_list]
                header_exception_list = [header_exception_list]

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(table[self._pnl_header], color="#003049", label=self._pnl_header)

        for i, head in enumerate(header_list):
            color = next(color_cycle)
            ax.plot(table[head], color=color, linestyle=next(line_style_cycle), alpha=0.7, label=head)

            exceed_0 = table[table[header_exception_list[i]] == True][self._pnl_header]

            ax.scatter(exceed_0.index, exceed_0, marker=next(marker_cycle), facecolors="none", edgecolors=color, s=120, label=header_exception_list[i])

        ax.spines["bottom"].set_color("#b0abab")
        ax.spines["top"].set_color("#b0abab")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.legend(loc="upper left", prop={"size": 12})

        ax.set_title(backtest_data.index.name + " VaR Backtest", fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def es_plot(
        self,
        backtest_data: pd.DataFrame,
        begin_date: str | None = None,
        end_date: str | None = None,
        alpha_index: int | slice | None = None,
    ) -> tuple[Figure, Axes]:
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
        fig, axes : Figure, Axes
            Matplotlib figure and axes object.
        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        header_list = []
        header_exception_list = []

        for windows in [self.len_alpha]:
            header_exception = self.header_exception[windows : windows + self.len_alpha]
            header = self.header[windows : windows + self.len_alpha]
            header_list.extend(header)
            header_exception_list.extend(header_exception)

        if alpha_index is not None:
            header_list = header_list[alpha_index]
            header_exception_list = header_exception_list[alpha_index]

            if isinstance(header_list, str):
                header_list = [header_list]
                header_exception_list = [header_exception_list]

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(table[self._pnl_header], color="#003049", label=self._pnl_header)

        for i, head in enumerate(header_list):
            color = next(color_cycle)
            ax.plot(table[head], color=color, linestyle=next(line_style_cycle), alpha=0.7, label=head)

            exceed_0 = table[table[header_exception_list[i]]][self._pnl_header]

            ax.scatter(exceed_0.index, exceed_0, marker=next(marker_cycle), facecolors="none", edgecolors=color, s=120)

        ax.spines["bottom"].set_color("#b0abab")
        ax.spines["top"].set_color("#b0abab")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.legend(loc="upper left", prop={"size": 12})

        ax.set_title(backtest_data.index.name + " Expected Shortfall Backtest", fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def dd_plot(
        self,
        backtest_data: pd.DataFrame,
        begin_date: str | None = None,
        end_date: str | None = None,
        alpha_index: int | slice | None = None,
    ) -> tuple[Figure, Axes]:
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
        fig, axes : Figure, Axes
            Matplotlib figure and axes object.
        """
        table = self.__get_data_range(backtest_data, begin_date, end_date)

        header_list = []
        header_exception_list = []

        for windows in [2 * self.len_alpha]:
            header_exception = self.header_exception[windows : windows + self.len_alpha]
            header = self.header[windows : windows + self.len_alpha]
            header_list.extend(header)
            header_exception_list.extend(header_exception)

        if alpha_index is not None:
            header_list = header_list[alpha_index]
            header_exception_list = header_exception_list[alpha_index]

            if isinstance(header_list, str):
                header_list = [header_list]
                header_exception_list = [header_exception_list]

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(table[self._pnl_header], color="#003049", label=self._pnl_header)

        for i, head in enumerate(header_list):
            color = next(color_cycle)
            ax.plot(table[head], color=color, linestyle=next(line_style_cycle), alpha=0.7, label=head)

            exceed_0 = table[table[header_exception_list[i]]][self._pnl_header]

            ax.scatter(exceed_0.index, exceed_0, marker=next(marker_cycle), facecolors="none", edgecolors=color, s=120, label=header_exception_list[i])

        ax.spines["bottom"].set_color("#b0abab")
        ax.spines["top"].set_color("#b0abab")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.legend(loc="upper left", prop={"size": 12})

        ax.set_title(backtest_data.index.name + " Conditional VaR Backtest", fontsize=16, fontweight=1)

        plt.tight_layout()
        plt.show()
