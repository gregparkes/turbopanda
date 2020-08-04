#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for intelligent scatterplots in primitive matplotlib."""

import numpy as np
import matplotlib as mpl
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import CategoricalDtype

from typing import Union, List, Tuple, Optional
from pandas import Series, Index
from warnings import warn

from turbopanda.utils import remove_na, instance_check,\
    arrays_equal_size, belongs, as_flattened_numpy, unique_ordered
from turbopanda.stats._kde import freedman_diaconis_bins

from ._default import _ArrayLike, _ListLike, _Numeric
from ._annotate import annotate
from ._palette import darken, lighten, \
    color_qualitative, cat_array_to_color
from ._widgets import map_legend


def _marker_set():
    return 'o', '^', 'x', 'd', '8', 'p', 'h', '+', 'v', '*'


def _select_best_size(x: int, a: float = 1.4, b: float = 21.) -> float:
    # given n points, determines which size of point to use based on math rule
    """Rule is : b - a*log(x) where best options found are b=21, a=1.4"""
    # cases=(1e1, 50, 1e2, 200, 500, 1e3, 3000, 1e4, 20000, 50000, 1e5)
    return b - a * np.log(x)


def _negsigmoid(x: float, a: float = .9) -> float:
    b = 1 - a
    return ((a - b) / (1. + np.exp(x))) + b


def _glf(x: float, a: float = 1., k: float = .2, c: float = 1.,
         b: float = 1.2, nu: float = 1., q: float = 1.) -> float:
    # generalized logistic function (see https://en.wikipedia.org/wiki/Generalised_logistic_function)
    return a + (k - a) / (c + q * np.exp(-b * x)) ** (1 / nu)


def _select_best_alpha(n: int,
                       glf_range: Tuple[float, float] = (-1., 9.),
                       measured_range: Tuple[float, float] = (1e1, 1e5)):
    # given n points, determines the best alpha to use
    """Follows a generalized logistic function Y(t)=A + (K - A) / (C + Q * exp(-B * t))**(1/nu)"""
    # clip x to the measured range
    _n = np.clip(n, *measured_range)
    # linearly interpolates x value into (-2, 8) range for generalized sigmoid-like function
    interp_n = np.interp(_n, measured_range, glf_range)
    return _glf(interp_n)


def _draw_line_best_fit(x, y, c, ax, deg):
    data_color = c if isinstance(c, str) else "k"
    # get line color as a bit darker
    if data_color != "k":
        line_color = darken(data_color)
    else:
        line_color = lighten(data_color)

    xln = 3 if deg == 1 else 200
    z = np.polyfit(x, y, deg=deg)
    xl = np.linspace(x.min(), x.max(), xln)
    ax.plot(xl, np.polyval(z, xl), color=line_color, linestyle="--")


def _make_colorbar(data, ax, cmap):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    norm = mpl.colors.Normalize(data.min(), data.max())
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if isinstance(data, Series):
        cbar.set_label(data.name)


def _draw_scatter(x, y, c, s, m, alpha, ax, **kwargs):
    if isinstance(m, str):
        return ax.scatter(x, y, c=c, s=s, marker=m, alpha=alpha, **kwargs)
    else:
        scatters = []
        names = unique_ordered(m)
        marks = tuple(it.islice(it.cycle(_marker_set()), 0, len(names)))
        for v, mark in zip(names, marks):
            # handle cases where c, s are standalone and/or arrays
            if isinstance(c, str) and isinstance(s, float):
                scatters.append(
                    ax.scatter(x[m == v], y[m == v],
                               c=c, s=s, alpha=alpha, marker=mark, **kwargs))
            elif (not isinstance(c, str)) and isinstance(s, float):
                scatters.append(
                    ax.scatter(x[m == v], y[m == v],
                               c=c[m == v], s=s, alpha=alpha, marker=mark, **kwargs))
            elif isinstance(c, str) and (not isinstance(s, float)):
                scatters.append(
                    ax.scatter(x[m == v], y[m == v],
                               c=c, s=s[m == v],
                               alpha=alpha, marker=mark, **kwargs))
            elif (not isinstance(c, str)) and (not isinstance(s, float)):
                scatters.append(ax.scatter(x[m == v], y[m == v],
                                           c=c[m == v], s=s[m == v],
                                           alpha=alpha, marker=mark, **kwargs))
        return scatters


def scatter(X: _ArrayLike,
            Y: _ArrayLike,
            c: Union[str, _ArrayLike] = 'k',
            marker: Union[str, _ArrayLike] = 'o',
            s: Optional[Union[_Numeric, _ArrayLike]] = None,
            dense: bool = False,
            fit_line: bool = False,
            ax: Optional[mpl.axes.Axes] = None,
            alpha: Optional[float] = None,
            cmap: str = "viridis",
            legend: bool = True,
            colorbar: bool = True,
            with_jitter: bool = False,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            x_scale: str = "linear",
            y_scale: str = "linear",
            legend_outside: bool = False,
            title: str = "",
            with_grid: bool = False,
            fit_line_degree: int = 1,
            **scatter_kws):
    """Generates a scatterplot, with some useful features added to it.

    Parameters
    ----------
    X : list/tuple/np.ndarray/pd.Series (1d)
        The data column to draw on the x-axis. Flattens if np.ndarray
    Y : list/tuple/np.ndarray/pd.Series (1d)
        The data column to draw on the y-axis. Flattens if np.ndarray
    c : str/list/tuple/np.ndarray/pd.Series (1d), default='blue'
        The colour of the points.
        If array, colors must be a categorical/valid float type, uses cmap
    marker : str/list/tuple/np.ndarray/pd.Series (1d), default='o'
        The marker style of the points.
        If type=list/array, array must be a categorical/str-like type to map to matplotlib markers
        If dense=True, treats each marker as a circle, ignores this input
    s : int/float/list/tuple/np.ndarray/pd.Series (1d), optional
        Size of each point.
        If dense=True, this value is set automatically.
        If type=list/array, array must be array of floats
    dense : bool
        If True, draws the uniform densities instead of the actual points
    fit_line : bool
        If True, draws a line of best fit on the data
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    alpha : float, optional
        Sets the alpha for colour. If dense is True, this value is set automatically
    cmap : str, default="viridis"
        The default colormap for continuous-valued c.
    legend : bool, default=True
        Draws a legend if the 'c' variable is discrete
    colorbar : bool, default=True
        Draws a colorbar if the 'c' variable is continuous
    with_jitter : bool, default=False
        If True, and dense=True, adds some jitter to the uniform points
    x_label : str, default="x-axis"
        If X is not a pandas.Series, this is used
    y_label : str, default="y-axis"
        If Y is not a pandas.Series, this is used
    x_scale : str, default="linear"
        Choose from {'linear', 'log', 'symlog', 'logit'}, see `matplotlib.ax.set_xscale`
    y_scale : str, default="linear"
        Choose from {'linear', 'log', 'symlog', 'logit'}, see `matplotlib.ax.set_yscale`
    legend_outside : bool, default=False
        If True, plots the legend outside the plot at (1, 1)
    title : str, default=""
        Optional title at the top of the axes
    with_grid : bool, default=False
        If True, draws a grid
    fit_line_degree : int, default=1
        If fit_line=True, Determines the degree to which a line is fitted to the data,
         allows polynomials

    Other Parameters
    ----------------
    scatter_kws : dict
        Keyword arguments to pass to `ax.scatter`

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-scatter
    """

    instance_check((X, Y), (list, tuple, np.ndarray, Series))
    instance_check((c, marker), (str, list, tuple, np.ndarray, Series, Index))
    instance_check(s, (type(None), int, float, list, tuple, np.ndarray, Series, Index))
    instance_check(alpha, (type(None), float))
    instance_check(ax, (type(None), mpl.axes.Axes))
    instance_check((dense, with_jitter, fit_line, with_grid, legend, legend_outside), bool)
    instance_check((x_label, y_label, title, x_scale, y_scale), (type(None), str))
    instance_check(fit_line_degree, int)

    arrays_equal_size(X, Y)
    if isinstance(marker, str):
        belongs(marker, _marker_set())

    # get subset where missing values from either are dropped
    _X = as_flattened_numpy(X)
    _Y = as_flattened_numpy(Y)

    # warn the user if n is large to maybe consider dense option?
    if _X.shape[0] > 15000 and not dense:
        warn("Data input size: {} is large, consider setting dense=True".format(X.shape[0]), UserWarning)

    # reconfigure colors if qualitative
    if isinstance(s, (list, tuple)) and not dense:
        s = as_flattened_numpy(s)
        arrays_equal_size(X, Y, s)
    if isinstance(marker, (list, tuple)) and not dense:
        marker = np.asarray(marker)
        arrays_equal_size(X, Y, marker)

    if not isinstance(c, str):
        # do some prep work on the color variable.
        palette, _cmode = cat_array_to_color(c, cmap=cmap)
        # perform size check
        arrays_equal_size(X, Y, palette)
    else:
        palette = c
        _cmode = "static"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if dense:
        # alpha, s are set in this case
        alpha = 0.8
        marker = "o"
        # perform density plotting
        bins_x = min(freedman_diaconis_bins(_X), 50)
        bins_y = min(freedman_diaconis_bins(_Y), 50)
        # estimate counts using histogram2d
        s, xs, ys = np.histogram2d(_X, _Y, bins=(bins_x, bins_y))
        # create a mesh
        xp, yp = np.meshgrid(xs[:-1], ys[:-1])
        if with_jitter:
            xp += np.random.rand(*xp.shape) / (_X.max() - _X.min())
            yp += np.random.rand(*yp.shape) / (_Y.max() - _Y.min())
    else:
        if alpha is None:
            alpha = _select_best_alpha(_X.shape[0])
        if s is None:
            s = _select_best_size(_X.shape[0])
        xp = _X
        yp = _Y

    # draw
    scat = _draw_scatter(xp, yp, palette, s, marker,
                         alpha, ax, cmap=cmap, **scatter_kws)

    # optionally fit a line of best fit
    if fit_line:
        _draw_line_best_fit(_X, _Y, palette, ax, fit_line_degree)

    if with_grid:
        ax.grid()

    # associate legend if colour map is used
    if _cmode == "discrete" and legend:
        map_legend(c, palette, marker, ax, legend_outside)
    elif _cmode == "continuous" and colorbar:
        # add colorbar
        _make_colorbar(c, ax, cmap)

    # apply x-label, y-label, title
    if isinstance(x_label, str):
        ax.set_xlabel(x_label)
    elif isinstance(X, Series):
        ax.set_xlabel(X.name)

    if isinstance(y_label, str):
        ax.set_ylabel(y_label)
    elif isinstance(Y, Series):
        ax.set_ylabel(Y.name)

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(title)

    return ax
