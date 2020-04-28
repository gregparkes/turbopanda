#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detects outliers in numerical samples."""
import numpy as np
from scipy import stats

from turbopanda.utils import as_flattened_numpy


def confidence_interval(data, confidence=.99):
    """Given data, calculate the percent confidence intervals for it.

    Returns mean, mean - ci, mean + ci
    """
    _a = as_flattened_numpy(data)
    return stats.t.interval(confidence, _a.shape[0] - 1, loc=np.mean(_a), scale=stats.sem(_a))
