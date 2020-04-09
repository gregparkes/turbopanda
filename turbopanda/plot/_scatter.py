#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for intelligent scatterplots in primitive matplotlib."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from pandas import Series

from turbopanda.stats import density
from turbopanda.utils import remove_na, instance_check
from turbopanda.stats._kde import freedman_diaconis_bins


def _select_best_size(x, a=1.4, b=21):
    # given n points, determines which size of point to use based on math rule
    """Rule is : b - a*log(x) where best options found are b=21, a=1.4"""
    # cases=(1e1, 50, 1e2, 200, 500, 1e3, 3000, 1e4, 20000, 50000, 1e5)
    return b - a * np.log(x)


def _negsigmoid(x, a=.9):
    b = 1 - a
    return ((a - b) / (1. + np.exp(x))) + b


def _glf(x, a=1., k=.2, c=1., b=1.2, nu=1., q=1.):
    # generalized logistic function (see https://en.wikipedia.org/wiki/Generalised_logistic_function)
    return a + (k - a) / (c + q * np.exp(-b * x)) ** (1 / nu)


def _select_best_alpha(n, glf_range=(-2, 8), measured_range=(1e1, 1e5)):
    # given n points, determines the best alpha to use
    """Follows a generalized logistic function Y(t)=A + (K - A) / (C + Q * exp(-B * t))**(1/nu)"""
    # clip x to the measured range
    _n = np.clip(n, *measured_range)
    # linearly interpolates x value into (-2, 8) range for generalized sigmoid-like function
    interp_n = np.interp(_n, measured_range, glf_range)
    return _glf(interp_n)


def scatter(X: Union[np.ndarray, Series, List, Tuple],
            Y: Union[np.ndarray, Series, List, Tuple],
            c: Union[str, np.ndarray, Series, List, Tuple] = 'blue',
            dense: bool = False,
            fit_line: bool = False,
            ax=None,
            alpha: Optional[float] = None,
            s: Optional[float] = None,
            with_jitter: bool = False,
            x_label: str = "x-axis",
            y_label: str = "y-axis",
            title: str = "",
            fit_line_degree: int = 1,
            **scatter_kws):
    """Generates a scatterplot, with some useful features added to it.

    Parameters
    ----------
    X : list/tuple/np.ndarray/pd.Series (1d)
        The data column to draw on the x-axis. Flattens if np.ndarray
    Y : list/tuple/np.ndarray/pd.Series (1d)
        The data column to draw on the y-axis. Flattens if np.ndarray
    c : str/np.ndarray/pd.Series (1d)
        The colour of the points
    dense : bool
        If True, draws the uniform densities instead of the actual points
    fit_line : bool
        If True, draws a line of best fit on the data
    ax : matplotlib.ax.Axes, optional, default=None
        If None, creates one.
    alpha : float, optional
        Sets the alpha for colour. If dense is True, this value is set automatically
    s : float, optional
        Size of the points. If dense is True, this value is set automatically
    with_jitter : bool, default=False
        If True, and dense=True, adds some jitter to the uniform points
    x_label : str, default="x-axis"
        If X is not a pandas.Series, this is used
    y_label : str, default="y-axis"
        If Y is not a pandas.Series, this is used
    title : str, default=""
        Optional title at the top of the axes
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
    instance_check(s, (type(None), int, float))
    instance_check(alpha, (type(None), float))
    instance_check(ax, (type(None), matplotlib.axes.Axes))
    instance_check((dense, with_jitter, fit_line), bool)
    instance_check((x_label, y_label, title), str)
    instance_check(fit_line_degree, int)

    # get subset where missing values from either are dropped
    _X, _Y = remove_na(np.asarray(X), np.asarray(Y), paired=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if dense:
        # alpha, s are set in this case
        alpha = 0.8
        # perform density plotting
        bins_x = min(freedman_diaconis_bins(_X), 50)
        bins_y = min(freedman_diaconis_bins(_Y), 50)
        # estimate counts using histogram2d
        s, xs, ys = np.histogram2d(_X, _Y, bins=(bins_x, bins_y))
        # create a mesh
        xp, yp = np.meshgrid(xs[:-1], ys[:-1])
        if with_jitter:
            xp += np.random.rand(xx.shape[0], xx.shape[1]) / (_X.max() - _X.min())
            yp += np.random.rand(yy.shape[0], yy.shape[1]) / (_Y.max() - _Y.min())
    else:
        if alpha is None:
            alpha = _select_best_alpha(_X.shape[0])
        if s is None:
            s = _select_best_size(_X.shape[0])
        xp = _X
        yp = _Y

    # draw
    ax.scatter(xp, yp, c=c, s=s, alpha=alpha, **scatter_kws)

    # maybe fit line?
    if fit_line:
        xln = 3 if fit_line_degree == 1 else 200
        z = np.polyfit(_X, _Y, deg=fit_line_degree)
        xl = np.linspace(_X.min(), _X.max(), xln)
        ax.plot(xl, np.polyval(z, xl), color=c, linestyle="--")

    # apply x-label, y-label, title
    if isinstance(X, Series):
        ax.set_xlabel(X.name)
    else:
        ax.set_xlabel(x_label)

    if isinstance(Y, Series):
        ax.set_ylabel(Y.name)
    else:
        ax.set_ylabel(y_label)

    ax.set_title(title)

    return ax
