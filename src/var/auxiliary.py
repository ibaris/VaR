# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Union

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from pandas._typing import Axes


__all__ = ["load_data", "number", "array_like", "data_frame", "number_like"]
__PATH__ = os.path.join(os.path.dirname(__file__), "data", "data.csv")

__DISTRIBUTIONS__ = {
    "cauchy": stats.cauchy,
    "chi2": stats.chi2,
    "expon": stats.expon,
    "exponpow": stats.exponpow,
    "gamma": stats.gamma,
    "lognorm": stats.lognorm,
    "norm": stats.norm,
    "powerlaw": stats.powerlaw,
    "rayleigh": stats.rayleigh,
    "uniform": stats.uniform,
    "t": stats.t,
    "gumbel_r": stats.gumbel_r,
    "f": stats.f,
    "laplace_asymmetric": stats.laplace_asymmetric,
}

number = Union[int, float]
number_like = Union[List[number], number]
array_like = Union[List[number], np.ndarray]
data_frame = pd.DataFrame
distributions = Literal[
    "chauchy",
    "chi2",
    "expon",
    "exponpow",
    "gamma",
    "lognorm",
    "norm",
    "powerlaw",
    "rayleigh",
    "uniform",
    "t",
    "gumbel_r",
    "f",
    "laplace_asymmetric",
]


def load_data():
    """
    A auxiliary function to load saved test data.

    Returns
    -------
    out : DataFrame
    """
    data = pd.read_csv(__PATH__)
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data = data.set_index("Date")

    return data / 100


@dataclass
class RiskResults:
    """
    RiskResults dataclass for storing risk metrics.

    Attributes
    ----------
    alpha : np.ndarray
        Array of confidence levels, shape (k,).
    var : np.ndarray
        Array of Value at Risk (VaR) values, shape (k,). Positive numbers representing loss units.
    es : np.ndarray
        Array of Expected Shortfall (ES) values, shape (k,). Positive numbers representing loss units.
    dd : np.ndarray or None, optional
        Array of Drawdown values, shape (k,). Positive numbers representing drawdown units. Defaults to None.
    """

    alpha: np.ndarray  # shape (k,)
    var: np.ndarray  # shape (k,), positive numbers (loss units)
    es: np.ndarray  # shape (k,), positive numbers (loss units)
    dd: np.ndarray | None = None  # shape (k,), positive numbers (drawdown units)
    name: str | None = None

    def __post_init__(self):
        headers = ["VaR", "ES", "DD"]

        self.header = []
        for i in range(len(headers)):
            self.header.extend([f"{headers[i]}({item * 100!s})" for item in 1 - self.alpha])

    @property
    def data(self):
        return np.concatenate((self.var, self.es, self.dd), axis=1)

    def to_df(self, index: Axes = None):
        data = self.data

        if index is None:
            index = [data.shape[0]]

        df = pd.DataFrame(data=data, columns=self.header, index=index)

        df.name = self.name

        return df
