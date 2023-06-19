# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Test Failure Rate
=================
*Created on 18.06.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

Test the failure rate of the VaR model with comparison to the failure rate computed by vartest.
"""
from var import VaR, load_data
import numpy as np
import vartests


def test_failure_rate():
    data = load_data()

    weights = np.array([0.40, 0.50, 0.10])
    var = VaR(data, weights)

    for method in var.methods:
        bth = var.backtest(method=method)
        evaluate = var.evaluate(bth)

        for i, item in enumerate(evaluate.index):
            bth_exp = bth[var.header_exception[i]]
            assert evaluate.loc[item, 'Percent'] == vartests.failure_rate(bth_exp)['failure rate']
