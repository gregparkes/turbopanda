#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to machine learning support in conjunction with MetaPanda.

The most basic beginning support we provide is automatic JSON caching of ML models.
"""

from ._fit_basic import fit_basic
from ._fit_grid import fit_grid, get_best_model

from ._plot import best_model_plot
from ._plot_overview import overview_plot, coefficient_plot
from ._plot_tune import parameter_tune_plot
from ._clean import ml_ready, make_polynomial
