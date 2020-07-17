#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty boxplots in primitive matplotlib using long or short form."""

from typing import Union, List, Tuple, Optional, Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from turbopanda.utils import instance_check, as_flattened_numpy, intersect, \
    difference, arrays_equal_size, listify, bounds_check
from turbopanda.str import shorten

from ._default import _ListLike, _ArrayLike
from ._palette import color_qualitative, contrast, noncontrast, autoshade


__all__ = ('box1d', 'bibox1d', 'widebox')


def _get_stars(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "-"


def _get_flier_style(style_name="white_circle"):
    """Presents a bunch of flier options.

    Choose from {'red_square', 'green_diamond'}
    """
    # a map for names of shapes to marker abbrievations
    markermap = {'square': 's', 'circle': 'o', 'diamond': 'D',
                 'point': '.', 'triangle': 'v', 'octagon': '8',
                 'pentagon': 'p', 'plus': '+', 'hexagon': 'h',
                 'cross': 'P', 'star': '*'}
    # split the string on _
    col, marker = style_name.split("_")
    return dict(markerfacecolor=col, marker=markermap[marker])


def _convert_x_scale(X, scale):
    # X is numpy.ndarray, scale is str or function
    if isinstance(scale, str):
        if not hasattr(np, scale):
            raise ValueError("scaling operation {} must be in {}".format(scale, scale_str_to_func))
        return getattr(np, scale)(X)
    elif callable(scale):
        return scale(X)
    else:
        raise TypeError("scale type {} not allowed, must be callable or str".format(type(scale)))


def _label_axes(ax, label, vert, rot, max_len):
    # if it's just one element then listify and perform
    _label = listify(label) if label is not None else [""]
    # go through and shorten words that are too long
    __label = shorten(_label, max_len)
    if vert:
        ax.set_xticklabels(__label)
    else:
        ax.set_yticklabels(__label)
    # do rotation if needed
    if not np.isclose(rot, 0.):
        if vert:
            ax.tick_params("x", rotation=rot)
            if rot < 0:
                for tick in ax.get_xmajorticklabels():
                    tick.set_horizontalalignment('left')
            else:
                for tick in ax.get_xmajorticklabels():
                    tick.set_horizontalalignment('right')
        else:
            ax.tick_params("y", rotation=rot)
            if rot < 0:
                for tick in ax.get_ymajorticklabels():
                    tick.set_horizontalalignment('left')
            else:
                for tick in ax.get_ymajorticklabels():
                    tick.set_horizontalalignment('right')


def _color_arrangement(ax, patch, color):
    # assign face color
    facecol = color_qualitative(1, False)[0] if color is None else color
    # define median colour
    sim_col = noncontrast(facecol)
    contr_col = contrast(facecol)
    for b in patch['boxes']:
        b.set_color(contr_col)
    for m in patch['medians']:
        m.set_color(contr_col)
    # set whisker/cap colour to be the contrast colour
    for w in patch['whiskers']:
        w.set_color(contr_col)
    for c in patch['caps']:
        c.set_color(contr_col)
    for b in patch['boxes']:
        b.set(facecolor=facecol)
    # set background of the axes to be the same-colour
    ax.set_facecolor(sim_col)
    return facecol


def _kcolor_arrangement(patch, color, k=2):
    # assign face color
    facecols = color_qualitative(k, False) if color is None else color
    for b, c in zip(patch['boxes'], facecols):
        b.set(facecolor=c)
    for m in patch['medians']:
        m.set_color("k")
    return facecols


def _define_boxplot_arguments(ax, patch, vert, measured,
                              grid, spines, capsize,
                              axis_scale):
    if measured is None:
        measured = "Observed values"
        if isinstance(axis_scale, str):
            measured = axis_scale + "(" + measured + ")"
    # draw bland axis
    if vert:
        ax.set_ylabel(measured)
    else:
        ax.set_xlabel(measured)
    # draw gridlines on appropriate axis
    if grid:
        if vert:
            ax.yaxis.grid(True)
        else:
            ax.xaxis.grid(True)
    # spine visibility
    all_spines = ('top', 'left', 'right', 'bottom')
    removed_spines = tuple(difference(all_spines, spines))
    for s in removed_spines:
        ax.spines[s].set_visible(False)
    # change capsize
    for cap in patch['caps']:
        cap.set_linewidth(capsize)
    # whisker length
    for w in patch['whiskers']:
        w.set_linewidth(1.2)


def _overlay_stripplot(X, ax, n, width, color, vert, outliers=True, strip_jitter=0.15):
    """Overlays a stripplot on top of a boxplot"""
    x_strip_color = autoshade(color, 0.15)
    x_strip_edge_color = contrast(x_strip_color)
    x_range = np.clip(np.full(X.shape[0], n) + np.random.normal(0, strip_jitter, size=X.shape[0]),
                      n - width, n + width)
    if not outliers:
        # calculate outliers
        _m = np.mean(X)
        _std3 = np.std(X) * 3.
        nonoutliers = (_m - _std3 < X) & (_m + _std3 > X)
        # select subset
        X = X[nonoutliers]
        x_range = x_range[nonoutliers]

    if vert:
        ax.scatter(x_range, X, alpha=.3, color=x_strip_color, edgecolors=x_strip_edge_color)
    else:
        ax.scatter(X, x_range, alpha=.3, color=x_strip_color, edgecolors=x_strip_edge_color)


def box1d(X: _ArrayLike,
          color: Optional[str] = None,
          label: Optional[str] = None,
          ax: Optional[mpl.axes.Axes] = None,
          with_strip: bool = False,
          vertical: bool = True,
          notch: bool = False,
          capsize: float = 1.0,
          outliers: bool = True,
          axis_scale: Optional[Union[str, Callable]] = None,
          grid: bool = True,
          width: float = 0.7,
          label_rotation: float = 0.,
          label_max_length: int = 25,
          spines: Optional[_ListLike] = None,
          theme: str = "white_circle",
          **plot_kwargs):
    """Plots a 1-dimensional boxplot using a vector.

    Parameters
    ----------
    X : list/tuple/np.ndarray/pd.Series (1d)
        The data column to draw. Must be numeric.
    color : str, optional
        If None, uses a default color
    label : str, optional
        If set, draws this on the appropriate axis, if None, does nothing
        If X is of type pandas.Series, uses this label instead.
    ax : matplotlib.ax object, optional, default=None
        If None, creates a plot.
    with_strip : bool, default=False
        If True, draws a stripplot over the top of the boxplot, in a similar colour
        `outliers` are set to False in this case
    vertical : bool, default=True
        Determines whether to draw the plot vertically or horizontally
    notch : bool, default=False
        Determines whether to draw a notched plot
    capsize : float, default=1.0
        Defines the length of the caps
    outliers : bool, default=True
        If True, displays outfliers as outliers
    axis_scale: str/callable, optional
        Scales the data along the axis.
        If str, use {'log', 'sqrt', 'log2'}
        If callable, must reference a `np.*` function which takes array X and returns X'
    grid : bool, default=True
        If True: draws gridlines for the numeric axis
    width : float, default=0.7
        Determines the width/height of the box
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    label_max_length : int, default=25
        If any label exceeds this length, it truncates it
    spines : tuple, default=('top','left',bottom','right')
        Defines which spines are to be visible
    theme : str, default="white_circle"
        Choose a 'theme' for the outliers, from {'red_square', 'green_diamond'}

    Other Parameters
    ----------------
    plot_kwargs : dict
        keyword arguments to pass to `ax.boxplot`

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-boxplot
    """

    instance_check(X, (np.ndarray, pd.Series, list, tuple))
    instance_check((vertical, notch, outliers, grid, with_strip), bool)
    instance_check(spines, (type(None), list))
    instance_check(theme, str)
    instance_check((label, color), (type(None), str))
    instance_check((capsize, width), float)
    instance_check(label_rotation, (int, float))
    instance_check(label_max_length, int)
    bounds_check(width, 0., 1.)

    # convert option to numpy
    _X = as_flattened_numpy(X)
    _style = _get_flier_style(theme)
    # convert X data if we have axis_scale
    if axis_scale:
        _X = _convert_x_scale(_X, axis_scale)

    if with_strip:
        outliers = False
    if ax is None and vertical:
        fig, ax = plt.subplots(figsize=(2.5, 5))
    elif ax is None and not vertical:
        fig, ax = plt.subplots(figsize=(5, 2.5))
    if spines is None:
        spines = ('left', 'top', 'right', 'bottom')
    box_alpha = 1. if not with_strip else .5

    patch_obj = ax.boxplot(_X, vert=vertical, patch_artist=True,
                           showfliers=outliers, notch=notch,
                           widths=width, boxprops=dict(alpha=box_alpha),
                           flierprops=_style, **plot_kwargs)
    # define basic arguments
    _define_boxplot_arguments(ax, patch_obj, vertical, None,
                              grid, spines, capsize, axis_scale)
    # define colour features
    color = _color_arrangement(ax, patch_obj, color)
    # label the appropriate axes
    _label_axes(ax, X.name if isinstance(X, pd.Series) else label,
                vertical, label_rotation, label_max_length)
    # plot the strips
    if with_strip:
        _overlay_stripplot(_X, ax, 1, width, color, vertical, outliers, strip_jitter=0.15)
    return ax


def bibox1d(X: _ArrayLike,
            Y: _ArrayLike,
            colors: Optional[_ListLike] = None,
            labels: Optional[_ListLike] = None,
            measured: Optional[str] = None,
            ax: Optional[mpl.axes.Axes] = None,
            mannwhitney: bool = True,
            with_strip: bool = False,
            vertical: bool = True,
            notch: bool = False,
            capsize: float = 1.0,
            outliers: bool = True,
            grid: bool = True,
            width: Union[float, List[float]] = 0.7,
            label_rotation: float = 0.0,
            label_max_length: int = 25,
            spines: Optional[_ListLike] = None,
            strip_jitter: float = 0.15,
            theme: str = "white_circle",
            **plot_kwargs):
    """Plots two 1-dimensional boxplots using vectors `X`, `Y`.

    Parameters
    ----------
    X : list/tuple/np.ndarray/pd.Series (1d)
        The first data column to draw. Must be numeric.
    Y : list/tuple/np.ndarray/pd.Series (1d)
        The second data column to draw. Must be numeric.
    colors : str/list of str, optional
        If None, uses a default color
    labels : str/list of str, optional
        If set, draws this on the appropriate axis, if None, does nothing
        If X/Y is of type pandas.Series, uses this label instead.
    measured : str, optional
        A label to define what the measurement is
    ax : matplotlib.ax object, optional, default=None
        If None, creates a plot.
    mannwhitney : bool, default=True
        If True, performs a Mann-Whitney U test between the values
    with_strip : bool, default=False
        If True, draws a stripplot over the top of the boxplot, in a similar colour
        `outliers` are set to False in this case
    vertical : bool, default=True
        Determines whether to draw the plot vertically or horizontally
    notch : bool, default=False
        Determines whether to draw a notched plot
    capsize : float, default=1.0
        Defines the length of the caps
    outliers : bool, default=True
        If True, displays fliers as outliers
    grid : bool, default=True
        If True: draws gridlines for the numeric axis
    width : float, default=0.7
        Determines the width/height of the box
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    label_max_length : int, default=25
        If any label exceeds this length, it truncates it
    spines : tuple, default=('top','left',bottom','right')
        Defines which spines are to be visible
    strip_jitter : float, default=0.15
        With stripplot, defines the amount of jitter in the variables
    theme : str, default="white_circle"
        Choose a 'theme' for the outliers, from {'red_square', 'green_diamond'}

    Other Parameters
    ----------------
    plot_kwargs : dict
        keyword arguments to pass to `ax.boxplot`

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-boxplot

    See Also
    --------
    matplotlib.pyplot.boxplot

    References
    ----------
    Inspiration from https://github.com/jbmouret/matplotlib_for_papers#colored-boxes
    """
    instance_check((X, Y), (list, tuple, np.ndarray, pd.Series))
    instance_check((colors, labels, spines), (type(None), list, pd.Index))
    instance_check(ax, (type(None), mpl.axes.Axes))
    instance_check((mannwhitney, vertical, notch, outliers, grid, with_strip), bool)
    instance_check((capsize, width, strip_jitter, label_rotation), (float, int))
    instance_check(theme, str)
    instance_check(label_max_length, int)
    bounds_check(strip_jitter, 0., 1.)

    _X = as_flattened_numpy(X)
    _Y = as_flattened_numpy(Y)
    _style = _get_flier_style(theme)

    if ax is None and vertical:
        fig, ax = plt.subplots(figsize=(3.5, 7))
    elif ax is None and not vertical:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    if with_strip:
        outliers = False

    if spines is None:
        if vertical and mannwhitney:
            spines = ('bottom', 'left', 'right')
        elif not vertical and mannwhitney:
            spines = ('bottom', 'left', 'top')
        else:
            spines = ('bottom', 'left', 'top', 'right')
    # sort out labels
    if labels is None:
        labels = [
            X.name if isinstance(X, pd.Series) else "",
            Y.name if isinstance(Y, pd.Series) else ""
        ]
    box_alpha = 1. if not with_strip else .5

    patch_obj = ax.boxplot(
        [_X, _Y], vert=vertical, patch_artist=True,
        showfliers=outliers, notch=notch,
        widths=width,
        flierprops=_style, boxprops=dict(alpha=box_alpha),
        **plot_kwargs
    )
    # define boxplot extras
    _define_boxplot_arguments(ax, patch_obj, vertical, measured,
                              grid, spines, capsize, None)
    # define basic colours - overrides if needs be
    colors = _kcolor_arrangement(patch_obj, colors)
    # label axes
    _label_axes(ax, labels, vertical, label_rotation, label_max_length)
    # if we have stripplot, draw this
    if with_strip:
        # plot x strips
        _overlay_stripplot(_X, ax, 1, width, colors[0], vertical, outliers, strip_jitter)
        _overlay_stripplot(_Y, ax, 2, width, colors[1], vertical, outliers, strip_jitter)

    # if we have mann-whitney append this info
    if mannwhitney:
        # determine mann-whitney U test
        z, p = mannwhitneyu(_X, _Y)
        # p-value * 2
        p *= 2
        star = _get_stars(p)
        # get dimensions to annotate
        joined = np.concatenate((_X, _Y))
        _max, _min = np.max(joined), np.min(joined)
        # annotate on mann-whitney test
        if vertical:
            ax.annotate("", xy=(1, _max), xycoords="data",
                        xytext=(2, _max), textcoords="data",
                        arrowprops=dict(arrowstyle="-", ec="#666666",
                                        connectionstyle="bar,fraction=0.2"))
            # add mw text
            ax.text(1.5, _max + np.abs(_max - _min) * .1, star,
                    horizontalalignment="center", verticalalignment="center")
        else:
            ax.annotate("", xy=(_max, 2), xycoords="data",
                        xytext=(_max, 1), textcoords="data",
                        arrowprops=dict(arrowstyle="-", ec="#666666",
                                        connectionstyle="bar,fraction=0.2"))
            # add mw text
            ax.text(_max + np.abs(_max - _min) * .1, 1.5, star,
                    horizontalalignment="center", verticalalignment="center")

    return ax


def widebox(data: Union[List, np.ndarray, pd.DataFrame],
            colors: Optional[_ListLike] = None,
            measured: Optional[str] = None,
            ax: Optional[mpl.axes.Axes] = None,
            vert: bool = True,
            sort: bool = True,
            outliers: bool = True,
            notch: bool = False,
            with_strip: bool = False,
            capsize: float = 1.0,
            width: float = 0.7,
            grid: bool = True,
            title: Optional[str] = None,
            label_rotation: float = 0.0,
            label_max_length: int = 25,
            spines: Optional[_ListLike] = None,
            strip_jitter: float = 0.15,
            theme="white_circle",
            **plot_kwargs):
    """Plots a 2D boxplot with data oriented in wide-form.

    Parameters
    ----------
    data : list, np.ndarray or pd.DataFrame (2d)
        The raw data to plot as a box.
        If data is of type pd.DataFrame: columns represent X-axis
    colors : list, tuple, optional
        Represents colors for each x-variable
    measured : str, optional
        A name for the measured variable
    ax : matplotlib.ax object, optional, default=None
        If None, creates a plot.
    vert : bool, default=True
        Determines whether to draw the plot vertically or horizontally
    sort : bool, default=True
        Determines whether to sort the data by numerical value
    outliers : bool, default=True
        If True, displays fliers as outliers
    notch : bool, default=False
        Determines whether to draw a notched plot
    with_strip : bool, default=False
        If True, draws a stripplot over the top of the boxplot, in a similar colour
        `outliers` are set to False in this case
    capsize : float, default=1.0
        Defines the length of the caps
    width : float, default=0.7
        Determines the width/height of the box
    grid : bool, default=True
        If True: draws gridlines for the numeric axis
    title : str, optional
        Sets the title of the axes if a string is passed
    label_rotation : float, default=0
        The degrees of rotation to the ticklabels
    label_max_length : int, default=25
        If any label exceeds this length, it truncates it
    spines : tuple, default=('top','left',bottom','right')
        Defines which spines are to be visible
    strip_jitter : float, default=0.15
        With stripplot, defines the amount of jitter in the variables
    theme : str, default="white_circle"
        Choose a 'theme' for the outliers, from {'red_square', 'green_diamond'}

    Other Parameters
    ----------------
    plot_kwargs : dict
        keyword arguments to pass to `ax.boxplot`

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-boxplot

    See Also
    --------
    matplotlib.pyplot.boxplot
    seaborn.boxplot
    seaborn.boxenplot

    References
    ----------
    Inspiration from https://github.com/jbmouret/matplotlib_for_papers#colored-boxes
    """
    instance_check(data, (list, np.ndarray, pd.DataFrame))
    instance_check((colors, spines), (type(None), list, pd.Index))
    instance_check(ax, (type(None), mpl.axes.Axes))
    instance_check((vert, sort, notch, outliers, grid, with_strip), bool)
    instance_check((capsize, width, strip_jitter, label_rotation), (float, int))
    instance_check(theme, str)
    instance_check(label_max_length, int)
    bounds_check(width, 0., 1.)
    bounds_check(strip_jitter, 0., 1.)
    bounds_check(label_rotation, 0., 360.)

    if isinstance(data, pd.DataFrame):
        # select float, int subset
        ss = data.select_dtypes(include=[float, int])
        _data = np.asarray(ss)
        _labels = ss.columns
    elif isinstance(data, (list, np.ndarray)):
        _data = np.asarray(data)
        _labels = None
    else:
        raise TypeError("data matrix is not of type np.ndarray")

    _style = _get_flier_style(theme)

    # negative-exponential increase in figure size with more features
    def _figure_spacing(x):
        return np.exp(-.35 * x) * x

    if with_strip:
        outliers = False
    if ax is None and vert:
        fig, ax = plt.subplots(figsize=(2.5 + _figure_spacing(_data.shape[1]), 7))
    elif ax is None and not vert:
        fig, ax = plt.subplots(figsize=(7, 2.5 + _figure_spacing(_data.shape[1])))
    if spines is None:
        spines = ('left', 'top', 'right', 'bottom')

    # sort the data by the mean if selected
    if sort:
        _order = np.argsort(np.mean(_data, axis=0))
        _data = _data[:, _order]
        _labels = _labels[_order]

    box_alpha = 1. if not with_strip else .5

    patch_obj = ax.boxplot(
        _data, vert=vert, patch_artist=True,
        widths=width, showfliers=outliers, notch=notch,
        flierprops=_style, boxprops=dict(alpha=box_alpha),
        **plot_kwargs
    )

    # define boxplot extras
    _define_boxplot_arguments(ax, patch_obj, vert, measured,
                              grid, spines, capsize, None)
    # define basic colours - overrides if needs be
    colors = _kcolor_arrangement(patch_obj, colors, k=_data.shape[1])
    # label axes
    _label_axes(ax, _labels, vert, label_rotation, label_max_length)
    if title is not None:
        ax.set_title(title)
    # perform stripplots
    if with_strip:
        for n in range(_data.shape[1]):
            # plot x strips
            _overlay_stripplot(_data[:, n], ax, n + 1,
                               width, colors[n], vert,
                               outliers, strip_jitter)

    return ax
