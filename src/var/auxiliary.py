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
from typing import List, Union

__all__ = ["load_data", "number", "array_like", "data_frame", "number_like"]
__PATH__ = os.path.join(os.path.dirname(__file__), "data", "data.csv")

number = Union[int, float]
number_like = Union[List[number], number]
array_like = Union[List[number], np.ndarray]
data_frame = pd.DataFrame


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
