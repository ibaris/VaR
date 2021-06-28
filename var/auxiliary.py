# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 28.06.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import os

import pandas as pd

__all__ = ["load_data"]
__PATH__ = os.path.join(os.path.dirname(__file__), "data", "data.csv")


def load_data():
    """
    A auxiliary function to load saved test data.

    Returns
    -------
    out : DataFrame
    """
    data = pd.read_csv(__PATH__)
    data = data.set_index("Date")
    return data
