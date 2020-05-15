#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for associating annotations to various matplotlib plots."""

from typing import Union, List, Tuple, Optional
import itertools as it
import numpy as np
import matplotlib as mpl
from pandas import Series, Index
import matplotlib.pyplot as plt

from turbopanda.utils import instance_check, arrays_equal_size, \
    as_flattened_numpy
from turbopanda.str import shorten


def annotate(X: Union[np.ndarray, Series, List, Tuple],
             Y: Union[np.ndarray, Series, List, Tuple],
             T: Union[np.ndarray, Series, List, Tuple],
             subset: Optional[Union[np.ndarray, Series, List, Tuple]] = None,
             ax: mpl.axes.Axes = None,
             word_shorten: Optional[int] = None,
             **annotate_kws):
    """Annotates a matplotlib plot with text.

    Offsets are pre-determined according to the scale of the plot.

    Parameters
    ----------
    X : list/tuple/np.ndarray/pd.Series (1d)
        The x-positions of the text.
    Y : list/tuple/np.ndarray/pd.Series (1d)
        The y-positions of the text.
    T : list/tuple/np.ndarray/pd.Series (1d)
        The text array.
    subset : list/tuple/np.ndarray/pd.Series (1d)
        An array of indices to select a subset from.
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    word_shorten : int, optional
        If not None, shortens annotated strings to be more concise and displayable

    Other Parameters
    ----------------
    annotate_kws : dict
        Other keywords to pass to `ax.annotate`

    Returns
    -------
    ax : matplotlib.ax.Axes
        The same matplotlib plot, or the one generated
    """
    instance_check((X, Y), (list, tuple, np.ndarray, Series))
    instance_check(T, (list, tuple, np.ndarray, Series, Index))
    instance_check(subset, (type(None), list, tuple, np.ndarray, Series, Index))
    instance_check(ax, (type(None), mpl.axes.Axes))
    arrays_equal_size(X, Y, T)
    # convert to numpy.
    _X = as_flattened_numpy(X)
    _Y = as_flattened_numpy(Y)
    _T = as_flattened_numpy(T)

    if word_shorten:
        _T = shorten(_T, newl=word_shorten)

    if _X.dtype.kind == 'f':
        _X += (_X.max() - _X.min()) / 30.
        _Y += -((_Y.max() - _Y.min()) / 30.)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if subset is None:
        for x, y, t in it.zip_longest(_X, _Y, _T):
            ax.annotate(t, xy=(x, y), **annotate_kws)
    else:
        for i in subset:
            ax.annotate(_T[i], xy=(_X[i], _Y[i]), **annotate_kws)

    return ax
