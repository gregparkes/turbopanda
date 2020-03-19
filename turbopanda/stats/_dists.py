#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defining different distributions."""

import string

from scipy import stats

__all__ = ('scipy_continuous_distributions', 'scipy_discrete_distributions')


def scipy_continuous_distributions():
    """Returns the scipy continuous distribution set."""
    return [s for s in dir(stats) if
            not s.startswith("_") and s[0] in string.ascii_lowercase and
            hasattr(getattr(stats, s), 'fit')]


def scipy_discrete_distributions():
    """Returns the scipy discrete distribution set."""
    return [s for s in dir(stats) if
            not s.startswith("_") and s[0] in string.ascii_lowercase and
            hasattr(getattr(stats, s), 'pmf')]
