#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to useful statistical functions."""

from ._dists import *
from ._kde import get_bins, univariate_kde
from ._linear_model import LinearModel
from ._matrix_check import is_invertible
from ._stats import *
