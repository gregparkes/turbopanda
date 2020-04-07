#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to machine learning support in conjunction with MetaPanda.

The most basic beginning support we provide is automatic JSON caching of ML models.
"""

from ._clean import make_polynomial, ml_ready
from ._pgrid import make_parameter_grid
from ._select_model import get_best_model
# global inputs
from . import plot, fit
