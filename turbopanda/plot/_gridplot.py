#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determines the most optimal shape for grid-like layouts of plots to take."""

from __future__ import absolute_import, division, print_function

import itertools as it
import string
import matplotlib.pyplot as plt
import numpy as np

from turbopanda.utils import belongs, nearest_factors, instance_check, nonnegative


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
             ax_size: int = 2,
             annotate_labels: bool = False,
             annotate_offset: float = 0.01):
    """Determines the most optimal shape for a set of plots.

    Parameters
    ----------
    n_plots : int
        The total number of plots.
    arrange : str, default="square"
        Choose from {'square', 'row' 'column'}. Indicates preference for direction of plots.
    ax_size : int, default=2
        The square size of each plot.
    annotate_labels : bool, default=False
        If True, adds A, B,.. K label to top-left corner of each axes.
    annotate_offset : float, default=0.01
        Determines the amount of offset for each label

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure
    axes : list of matplotlib.ax.Axes
        A list of axes to use.
    """
    nonnegative((n_plots, ax_size,), int)
    belongs(arrange, ['square', 'row', 'column'])

    annot_props = {'weight': 'bold', 'horizontalalignment': 'left',
                   'verticalalignment': 'center'}

    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(ax_size, ax_size))  #
        # wrap ax as a list to iterate over.
        if annotate_labels:
            fig.text(0.01, .98, "A", **annot_props)
        return fig, [ax]
    else:
        fig, ax = _generate_square_like_grid(n_plots, ax_size=ax_size) \
            if arrange == 'square' else _generate_diag_like_grid(n_plots, arrange, ax_size=ax_size)
        # add annotation labels, hmmm
        if annotate_labels:
            # we use tight layout to make sure text isnt overlapping
            fig.tight_layout()
            for a, n in zip(ax, string.ascii_uppercase):
                pos_ = a.get_position().bounds
                # add label
                fig.text(pos_[0] - annotate_offset,
                         pos_[1] + pos_[3] + annotate_offset,
                         n, **annot_props)
        return fig, ax
