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

import numpy as np
from tqdm import tqdm

# import vartests
from var import VaR, load_data


def test_failure_rate():
    data = load_data()

    weights = np.array([0.40, 0.50, 0.10])
    var = VaR(data, weights)

    # var.fit_distributions()
    # var.historic()
    # var.parametric()
    # var.monte_carlo()
    bth = var.backtest(method="h")
    evaluate = var.evaluate(bth)

    # for method in tqdm(var.methods, desc="Testing Backtest"):
    #     bth = var.backtest(method=method)
    #     evaluate = var.evaluate(bth)


data = load_data()

weights = np.array([0.40, 0.50, 0.10])
var = VaR(data, weights)
var.fit_distributions(distribution="gamma")

# var.fit_distributions()
# var.historic()
# var.parametric()
# var.monte_carlo()
bth = var.backtest(method="g")
evaluate = var.evaluate(bth)

var.es_plot(bth)
