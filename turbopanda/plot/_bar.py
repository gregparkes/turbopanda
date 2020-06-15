#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty barplots."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple

from turbopanda.utils import as_flattened_numpy, instance_check, belongs
from turbopanda._deprecator import unimplemented

from ._palette import palette_cmap, convert_categories_to_colors, lighten
from ._annotate import annotate as annotate_labels
from ._default import _Numeric, _ArrayLike, _ListLike
from ._widgets import map_legend

__all__ = ('bar1d', 'widebar', 'errorbar1d')


def _plot_bar_vert(ax, ticks, labels, values, sd=None, c='k', w=0.8,
                   lrot=0., annotate=False, lines=None, vlabel=None):
    bar_args = {'ecolor': 'k', 'capsize': 4}
    # plot bar here
    if sd is not None:
        err_c = lighten(c, .2)
        ax.bar(ticks, values, yerr=sd, width=w, color=err_c, **bar_args)
    else:
        ax.bar(ticks, values, width=w, color=c)
    # set ticks and labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=lrot)
    # add optional horizontal lines
    if lines:
        ax.hlines(lines, xmin=-.5, xmax=ticks.shape[0] - .5, linestyle="--", color="k")
    # add optional annotations
    if annotate:
        annotate_labels(ticks, values, values, ax=ax)
    # set the value label
    ax.set_ylabel(vlabel)


def _plot_bar_hoz(ax, ticks, labels, values, sd=None, c='k', w=0.8,
                  lrot=0., annotate=False, lines=None, vlabel=None):
    # plot vbar
    bar_args = {'ecolor': 'k', 'capsize': 4}
    if sd is not None:
        err_c = lighten(c, .2)
        ax.barh(ticks, values, xerr=sd, height=w, color=err_c, **bar_args)
    else:
        ax.barh(ticks, values, height=w, color=c)
    # set ticks and labels
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=lrot)
    # add optional vertical lines
    if lines:
        ax.vlines(lines, ymin=-.5, ymax=ticks.shape[0] - .5, linestyle="--", color='k')
    # add optional annotations
    if annotate:
        annotate_labels(values, ticks, values, ax=ax)
    # set the value label
    ax.set_xlabel(vlabel)


def _determine_color_palette(cval, nticks, cmap):
    if cval is None:
        return palette_cmap(nticks, cmap=cmap)
    elif not isinstance(cval, str):
        pal, _ = convert_categories_to_colors(np.asarray(cval), cmap=cmap)
        return pal
    else:
        return cval


def _apply_data_sort(order, *arrays):
    # sort each array and return the complete list
    return list(map(lambda x: x[order], arrays))


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
    belongs(sort_by, ('values', 'labels'))

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
    pal = _determine_color_palette(c, _ticks.shape[0], cmap)

    # perform sorting here
    if sort:
        if sort_by == "values":
            _order = np.argsort(_values)
        elif sort_by == "labels":
            _order = np.argsort(_labels)
        else:
            raise ValueError("sort_by '{}': must be ['values', 'labels']".format(sort_by))
        # apply sort
        if not isinstance(c, (type(None), str)):
            _labels, _values, pal = _apply_data_sort(_order, _labels, _values, pal)
        else:
            _labels, _values = _apply_data_sort(_order, _labels, _values)

    if vert:
        _plot_bar_vert(ax, _ticks, _labels, _values, c=pal, w=width, lrot=label_rotation,
                       annotate=annotate, lines=vlinesAt, vlabel=value_label)
    else:
        _plot_bar_hoz(ax, _ticks, _labels, _values, c=pal, w=width, lrot=label_rotation,
                      annotate=annotate, lines=hlinesAt, vlabel=value_label)

    # map a legend to it
    if legend and not isinstance(c, str):
        map_legend(c, pal, 'o', ax, False)

    return ax


def errorbar1d(data: pd.DataFrame,
               c: Optional[Union[_ArrayLike, str]] = 'k',
               axis: Union[str, int] = 1,
               vert: bool = True,
               sort: bool = True,
               ax: Optional[mpl.axes.Axes] = None,
               width: float = 0.8,
               label_rotation: float = 0.0,
               cmap: str = "Blues"):
    """Plots a barplot with error bars.

    Data is to be arranged such that one axis is the categorical variables,
        and the other is the axis to aggregate along (using mean and std)

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot on.
    c : str/list/tuple/np.ndarray/pd.Series (1d), optional
        Defines the colour of each bar.
        If str, colours all of the bars with the same
        If array, must be a categorical type.
        If None, uses an automatic qualitative palette
    axis : int or str, default=1
        Choose from {0, 'rows', 1, 'columns'} to aggregate on.
    vert : bool, default=True
        Determines whether the plot is vertical or horizontal
    sort : bool, default=True
        Sorts the data or labels
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    width : float, default=0.8
        The width of each bar in the barplot
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    cmap : str, default="Blues"
        Defines a colormap if color values are specified

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-boxplot
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # fetch ticks and labels based on the opposite axis
    _labels = data.index if axis in ('columns', 1) else data.columns
    _ticks = np.arange(len(_labels))
    # given axis, get mean and std values to plot.
    m_v = data.mean(axis=axis).values
    m_sd = data.std(axis=axis).values
    # sort out colour
    pal = _determine_color_palette(c, _ticks.shape[0], cmap)

    # perform sorting if true
    if sort:
        _ord = np.argsort(m_v)
        # apply order to data
        if not isinstance(c, (type(None), str)):
            _labels, m_v, m_sd, pal = _apply_data_sort(_ord, _labels, m_v, m_sd, pal)
        else:
            _labels, m_v, m_sd = _apply_data_sort(_ord, _labels, m_v, m_sd)

    # now plot
    if vert:
        _plot_bar_vert(ax, _ticks, _labels, m_v, m_sd, c=pal,
                       w=width, lrot=label_rotation)
    else:
        _plot_bar_hoz(ax, _ticks, _labels, m_v, m_sd, c=pal,
                      w=width, lrot=label_rotation)

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
