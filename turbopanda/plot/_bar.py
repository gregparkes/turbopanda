#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty barplots."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple

from turbopanda.utils import as_flattened_numpy, instance_check
from turbopanda._deprecator import unimplemented

from ._palette import palette_cmap, convert_categories_to_colors
from ._annotate import annotate as annotate_labels
from ._default import _Numeric, _ArrayLike, _ListLike
from ._widgets import map_legend


def bar1d(X: _ArrayLike,
          Y: Optional[_ListLike] = None,
          c: Optional[Union[_ArrayLike, str]] = 'k',
          vert: bool = True,
          sort: bool = True,
          ax: Optional[mpl.axes.Axes] = None,
          annotate: bool = False,
          legend: bool = False,
          width: float = 0.8,
          label_rotation: float = 0.0,
          value_label: Optional[str] = None,
          sort_by: str = "values",
          cmap: str = "Blues",
          vlinesAt: Optional[Union[_Numeric, _ListLike]] = None,
          hlinesAt: Optional[Union[_Numeric, _ListLike]] = None):
    """Plots a 1 dimensional barplot.

    Parameters
    ----------
    X : list, tuple, np.ndarray, pd.Series
        Categorical/string/time labels for the data.
        If pandas.Series, Index must be categorical, Values must be numeric.
    Y : list, tuple, np.ndarray, optional
        If None, X must be a pd.Series. Must be numeric dtype.
    c : str/list/tuple/np.ndarray/pd.Series (1d), optional
        Defines the colour of each bar.
        If str, colours all of the bars with the same
        If array, must be a categorical type.
        If None, uses an automatic qualitative palette
    vert : bool, default=True
        Determines whether the plot is vertical or horizontal
    sort : bool, default=True
        Sorts the data or labels
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    annotate : bool, default=False
        Determines whether values should be annotated
    legend : bool, default=False
        Choose whether to display a legend
    width : float, default=0.8
        The width of each bar in the barplot
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    value_label : str, optional
        Defines a name for the numerical axis
    sort_by : str, default="values"
        Defines how to sort the data if sort=True.
        Choose from {'values', 'labels'}
    cmap : str, default="Blues"
        Defines a colormap if color values are specified
    vlinesAt : int, float, list, tuple, optional
        If set, defines one or more vertical lines to add to the barplot
    hlinesAt : int, float, list, tuple, optional
        If set, defines one or more horizontal lines to add to the barplot

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-boxplot

    """
    # define plot if not set
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # X is either numerical (in which case there is no Y, or categorical labels)
    if Y is None:
        # in this case, X must contain all the data
        if isinstance(X, pd.Series):
            _labels = as_flattened_numpy(X.index)
            _values = as_flattened_numpy(X.values)
            _ticks = np.arange(X.shape[0])
            value_label = X.name
        else:
            _labels = _ticks = np.arange(len(X))
            _values = as_flattened_numpy(X)
    else:
        # X is labels, Y are numeric values (assume!)
        _labels = as_flattened_numpy(X)
        _values = as_flattened_numpy(Y)
        _ticks = np.arange(_labels.shape[0])

    # sort out colour
    if c is None:
        pal = palette_cmap(_ticks.shape[0], cmap=cmap)
    elif not isinstance(c, str):
        pal, _ = convert_categories_to_colors(np.asarray(c), cmap=cmap)
    else:
        pal = c

    # perform sorting here
    if sort:
        if sort_by == "values":
            _order = np.argsort(_values)
        elif sort_by == "labels":
            _order = np.argsort(_labels)
        else:
            raise ValueError("sort_by '{}': must be ['value', 'alphabet']".format(sort_by))
        # apply sort
        _labels = _labels[_order]
        _values = _values[_order]
        # apply sort to colour if its a list-form
        if not isinstance(c, str):
            pal = pal[_order]

    if vert:
        ax.bar(_ticks, _values, width=width, color=pal)
        ax.set_xticks(_ticks)
        ax.set_xticklabels(_labels, rotation=label_rotation)
        if hlinesAt:
            ax.hlines(hlinesAt, xmin=-.5, xmax=_ticks.shape[0] - .5, linestyle="--", color="k")
        # annotations
        if annotate:
            annotate_labels(_ticks, _values, _values, ax=ax)
        ax.set_ylabel(value_label)
    else:
        ax.barh(_ticks, _values, height=width, color=pal)
        ax.set_yticks(_ticks)
        ax.set_yticklabels(_labels, rotation=label_rotation)
        if vlinesAt:
            ax.vlines(vlinesAt, ymin=-.5, ymax=_ticks.shape[0] - .5, linestyle="--", color='k')
        # annotate
        if annotate:
            annotate_labels(_values, _ticks, _values, ax=ax)
        ax.set_xlabel(value_label)

    # map a legend to it
    if legend and not isinstance(c, str):
        map_legend(c, pal, 'o', ax, False)

    return ax


@unimplemented
def widebar(data: pd.DataFrame,
            X: Optional[str] = None,
            Y: Optional[Union[str, _ListLike]] = None,
            c: Optional[Union[_ArrayLike, str]] = 'k',
            vert: bool = True,
            sort: bool = True,
            ax: Optional[mpl.axes.Axes] = None,
            total_width: float = 0.9):
    """Plots a barplot with hues.

    Note that columns in the data correspond to data that is to be 'hued'.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    _n, _p = data.shape
    if X is None:
        _labels = data.index
    else:
        # assumes x must be a column name
        _labels = data[x]
    # modify _p based on the size of y if given
    if Y is not None:
        _values = data[Y]
        _p = 1 if isinstance(Y, str) else len(Y)
    else:
        if X is not None:
            _values = data.drop(X, axis=1)
            _p -= 1
        else:
            _values = data

    _ticks = np.arange(_n)
    prop_tick = (1. / _p) - (1. - total_width)
    # toy palette
    pal = palette_cmap(_p, "Reds")
