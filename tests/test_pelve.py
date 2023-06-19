# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Test PELVE
==========
*Created on 20.06.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

Test the PELVE methods in the VaR class.
"""

import numpy as np
from var import VaR, load_data, objectives

data = load_data()
weights = np.array([0.30, 0.60, 0.10])
var = VaR(data, weights)


def test_pelve_parametric_objective():
    var_value = var.parametric()["VaR(99.0)"].values[0]
    objective = objectives.pelve_parameteric(var_value, var.pnl.values.flatten(), var.info["Portfolio Volatility"])
    test_1 = objective(0.01)
    target_1 = 0.027561900353827606

    test_2 = objective(
        0.075
    )  # This means, that the ES value at a 92.5% confidence level is approximately the same as the VaR value at 99% confidence level.
    target_2 = 0.00033400209647668766

    assert np.isclose(test_1, target_1)
    assert np.isclose(test_2, target_2)


def test_pelve_historic_objective():
    var_value = var.historic()["VaR(99.0)"].values[0]
    objective = objectives.pelve_historic(var_value, var.pnl.values.flatten())
    test_1 = objective(0.01)
    target_1 = 0.019138841666666684

    test_2 = objective(0.075)
    target_2 = 0.03233806723404255

    assert np.isclose(test_1, target_1)
    assert np.isclose(test_2, target_2)
