#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the patsy interface for matrix design."""

from typing import Optional

from turbopanda.utils import intersect


def patsy(self, X, y: Optional[str] = None):
    """Creates a `patsy`-compatible design matrix string.

    Parameters
    ----------
    X : selector
        The input/exogenous columns to select for.
    y : str, optional, default=None
        The name of the response variable

    Returns
    -------
    s : str
        Design matrix and y string
    """
    if isinstance(X, str):
        _x = self.select(X)
    else:
        _x = self.view(X)
    # continuous x variables.
    _x_cont = " + ".join(intersect(_x, self.view(float)))
    # use C brackets to denote categoricals.
    _x_dis = "C(" + ") + C(".join(intersect(_x, self.view_not(float))) + ")"
    _x_full = _x_cont + " + " + _x_dis

    s = _x_full
    if y is not None:
        s = y + " ~ " + _x_full

    return s
