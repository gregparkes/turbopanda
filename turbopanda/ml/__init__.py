#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to machine learning support in conjunction with MetaPanda.

The most basic beginning support we provide is automatic JSON caching of ML models.
"""

from ._clean import make_polynomial, ml_ready
from ._fit_basic import fit_basic
from ._fit_grid import fit_grid, get_best_model
from ._fit_learning import fit_learning
from ._plot import best_model_plot
from ._plot_learning import learning_curve_plot
from ._plot_overview import coefficient_plot, overview_plot
from ._plot_tune import parameter_tune_plot
