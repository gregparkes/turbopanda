#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to useful statistical functions."""

from ._kde import get_bins, univariate_kde, freedman_diaconis_bins
from ._dists import auto_fit
from ._density import density
from ._linear_model import LinearModel
from ._stats import *
from ._lmfast import lm
