# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Test VaR Methods
================
*Created on 19.06.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

Test the Value at Risk methods in the methods module.
"""

from var.methods import *
import numpy as np
from scipy.stats import norm


def test_parametric():
    # Assume some test data and expected results
    test_pnl = np.array([10, -5, 20, -15, 30, -10, 15, -5, 25, -15])
    test_alpha = np.array([0.05, 0.01])  # Example alpha values
    daily_std = np.std(test_pnl)
    expected_results = np.array([-21.5224678, -32.51123235, -21.5224678, -32.51123235, 19., 19.])

    # Call the function with the test data
    results = parametric(test_pnl, test_alpha, daily_std)

    # Test whether the results are as expected
    np.testing.assert_array_almost_equal(results, expected_results, decimal=4)


def test_historic():
    # Assume some test data and expected results
    test_pnl = np.array([10, -5, 20, -15, 30, -10, 15, -5, 25, -15])
    test_alpha = np.array([0.05, 0.01])  # Example alpha values
    expected_results = np.array([-15, -15, -15, -15, 19, 19])

    # Call the function with the test data
    results = historic(test_pnl, test_alpha)

    # Test whether the results are as expected
    np.testing.assert_array_almost_equal(results, expected_results, decimal=5)
