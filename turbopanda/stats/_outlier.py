#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detects outliers in numerical samples."""
import numpy as np
from scipy import stats


def confidence_interval(data, confidence=.99):
    """Given data, calculate the percent confidence intervals for it.

    Returns mean, mean - ci, mean + ci
    """
    _a = np.asarray(data).flatten()
    return stats.t.interval(confidence, _a.shape[0] - 1, loc=np.mean(_a), scale=stats.sem(_a))
