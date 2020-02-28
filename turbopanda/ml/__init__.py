#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to machine learning support in conjunction with MetaPanda.

The most basic beginning support we provide is automatic JSON caching of ML models.
"""

from ._fit_basic import fit_basic
from ._fit import fit_grid
from ._plot import *
from ._plot_overview import overview_plot, coefficient_plot
from ._clean import ml_ready
