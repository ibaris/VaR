# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import os

import pandas as pd
import numpy as np
from typing import List, Union, Literal
from scipy import stats

__all__ = ["load_data", "number", "array_like", "data_frame", "number_like"]
__PATH__ = os.path.join(os.path.dirname(__file__), "data", "data.csv")

__DISTRIBUTIONS__ = {
    'cauchy': stats.cauchy,
    'chi2': stats.chi2,
    'expon': stats.expon,
    'exponpow': stats.exponpow,
    'gamma': stats.gamma,
    'lognorm': stats.lognorm,
    'norm': stats.norm,
    'powerlaw': stats.powerlaw,
    'rayleigh': stats.rayleigh,
    'uniform': stats.uniform,
    't': stats.t,
    "gumbel_r": stats.gumbel_r,
    "f": stats.f
}

number = Union[int, float]
number_like = Union[List[number], number]
array_like = Union[List[number], np.ndarray]
data_frame = pd.DataFrame
distributions = Literal["chauchy", "chi2", "expon", "exponpow", "gamma", "lognorm", "norm", "powerlaw", "rayleigh",
                        "uniform", "t", "gumbel_r", "f"]


def load_data():
    """
    A auxiliary function to load saved test data.

    Returns
    -------
    out : DataFrame
    """
    data = pd.read_csv(__PATH__)
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
    data = data.set_index("Date")

    return data / 100
