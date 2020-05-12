#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty barplots."""

import itertools as it
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from turbopanda._deprecator import unimplemented
from turbopanda.utils import as_flattened_numpy, instance_check


@unimplemented
def bar1d(X, Y=None, c="b", vert=True, ax=None, label_rotation=0.):
    """Plots a 1 dimensional barplot.

    Parameters
    ----------
    X : list, tuple, np.ndarray, pd.Series
        Categorical/string/time labels for the data.
        If pandas.Series, Index must be categorical, Values must be numeric.
        If pandas.DataFrame, treats Index as categorical, each column as a hue
            for the barplot. Ignores Y.
    Y : list, tuple, np.ndarray, optional
        If None, X must be a pd.Series. Must be numeric dtype.
    c : str/list/tuple/np.ndarray/pd.Series (1d), default='blue'
        The colour of each bar.
        If array, must be a categorical type.
    vert : bool, default=True
        Determines whether the plot is vertical or horizontal
    width : float, default=0.8
        The width of each bar in the barplot
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    """
    instance_check(X, (list, tuple, np.ndarray, pd.Series, pd.Index))
    instance_check(Y, (type(None), list, tuple, np.ndarray))
    instance_check(vert, bool)
    instance_check(ax, (type(None), matplotlib.axes.Axes))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    # if we just have X and X is a pandas.Series, we dissect this into subparts
    if isinstance(X, pd.Series) and Y is None:
        # handle this case
        _cat = np.arange(X.shape[0])
        _X = as_flattened_numpy(X.index)
        _Y = as_flattened_numpy(X.values)
    else:
        _X = as_flattened_numpy(X)
        _Y = as_flattened_numpy(Y)
        _cat = np.arange(X.shape[0])

    if vert:
        ax.bar(_cat, _Y, width=width, color=c)
        ax.set_xticks(_cat)
        ax.set_xticklabels(_X, rotation=label_rotation)
    else:
        ax.barh(_cat, _Y, height=width, color=c)
        ax.set_yticks(_cat)
        ax.set_yticklabels(_X, rotation=label_rotation)
