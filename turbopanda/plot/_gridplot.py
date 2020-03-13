#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determines the most optimal shape for grid-like layouts of plots to take."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import matplotlib.pyplot as plt

from turbopanda.utils import nearest_factors, belongs


def _clean_axes_objects(n, axes):
    """Given axes and true n, clean and hide redundant axes plots."""
    if axes.ndim > 1:
        # flatten to 1d
        axes = list(it.chain.from_iterable(axes))
    if len(axes) > n:
        # if we have too many axes
        for i in range(len(axes) - n):
            # set invisible
            axes[np.negative(i + 1)].set_visible(False)
    return axes


def _generate_square_like_grid(n, ax_size=2):
    """
    Given n, returns a fig, ax pairing of square-like grid objects
    """
    f1, f2 = nearest_factors(n, shape="square")
    fig, axes = plt.subplots(ncols=f1, nrows=f2, figsize=(ax_size * f1, ax_size * f2))
    # update axes with clean
    axes = _clean_axes_objects(n, axes)
    return fig, axes


def _generate_diag_like_grid(n, direction, ax_size=2):
    """ Direction is in [row, column]"""
    belongs(direction, ["row", "column"])

    f1, f2 = nearest_factors(n, shape="diag")
    fmax, fmin = max(f1, f2), min(f1, f2)
    # get longest one
    tup, nc, nr = ((ax_size * fmin, ax_size * fmax), fmin, fmax) \
        if direction == 'row' else ((ax_size * fmax, ax_size * fmin), fmax, fmin)
    fig, axes = plt.subplots(ncols=nc, nrows=nr, figsize=tup)
    axes = _clean_axes_objects(n, axes)
    return fig, axes


def gridplot(n_plots: int,
             arrange: str = "square",
             ax_size: int = 2):
    """Determines the most optimal shape for a set of plots.

    Parameters
    ----------
    n_plots : int
        The total number of plots.
    arrange : str
        Choose from {'square', 'row' 'column'}. Indicates preference for direction
        of plots.
    ax_size : int
        The square size of each plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure
    axes : list of matplotlib.ax.Axes
        A list of axes to use.
    """
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(ax_size, ax_size))  #
        # wrap ax as a list to iterate over.
        return fig, [ax]
    else:
        return _generate_square_like_grid(n_plots, ax_size=ax_size) \
            if arrange == 'square' else _generate_diag_like_grid(n_plots, arrange, ax_size=ax_size)
