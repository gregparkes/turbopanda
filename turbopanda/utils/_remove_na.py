#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for removing missing values."""

import numpy as np


def _remove_na_single(x, axis="rows"):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def remove_na(x: np.ndarray, y: np.ndarray = None, paired=False, axis="rows"):
    """Remove missing values along a given axis in one or more (paired) numpy
    arrays.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    x, y : np.ndarray, array like
        Data. ``x`` and ``y`` must have the same number of dimensions.
        ``y`` can be None to only remove missing values in ``x``.
    paired : bool
        Indicates if the measurements are paired or not.
    axis : str
        Axis or axes along which missing values are removed.
        Can be 'rows' or 'columns'. This has no effect if ``x`` and ``y`` are
        one-dimensional arrays.

    Returns
    -------
    x, y : np.ndarray
        Data without missing values
    """
    # Safety checks
    x = np.asarray(x)
    assert x.size > 1, "x must have more than one element."
    assert axis in ["rows", "columns"], "axis must be rows or columns."

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    elif isinstance(y, (list, np.ndarray)):
        y = np.asarray(y)
        # Make sure that we just pass-through if y have only 1 element
        if y.size == 1:
            return _remove_na_single(x, axis=axis), y
        if x.ndim != y.ndim or paired is False:
            # x and y do not have the same dimension
            x_no_nan = _remove_na_single(x, axis=axis)
            y_no_nan = _remove_na_single(y, axis=axis)
            return x_no_nan, y_no_nan

    # At this point, we assume that x and y are paired and have same dimensions
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
    else:
        # 2D arrays
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y
