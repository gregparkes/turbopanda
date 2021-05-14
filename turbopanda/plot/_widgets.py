#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creates custom legends."""

import itertools as it
import pandas as pd
from matplotlib.lines import Line2D

from turbopanda.utils import is_twotuple, unique_ordered, zipe


def _marker_set():
    return "o", "^", "x", "d", "8", "p", "h", "+", "v", "*"


def legend_line(group_to_colors, lw=3.0):
    """
    Given a mapping of grouped-labels to RGB, create a legend object to be handled.

    Parameters
    -------
    group_to_colors : dict/2-tuple/pd.Index/pd.Series
        k (group names), v (RGB or hex values)
    lw : float
        The default line width for each legend

    Returns
    ------
    clines : list of Line2D
        Objects to pass into ax.legend when overriding legend.
    """
    if isinstance(group_to_colors, dict):
        custom_lines = [
            Line2D([0], [0], color=colour, lw=lw)
            for label, colour in group_to_colors.items()
        ]
    elif isinstance(group_to_colors, (list, tuple)):
        is_twotuple(group_to_colors)
        custom_lines = [
            Line2D([0], [0], color=colour, lw=lw) for label, colour in group_to_colors
        ]
    elif isinstance(group_to_colors, (pd.Index, pd.Series)):
        custom_lines = [
            Line2D([0], [0], color=colour, lw=lw) for colour in group_to_colors.values
        ]
    else:
        raise TypeError(
            "type '{}' not recognized for `legend`".format(type(group_to_colors))
        )
    return custom_lines


def legend_scatter(markers, colors, labels, msize=15):
    """Given a mapping of markers, colors and labels, returns a scatterplot legend."""
    custom_scats = [
        Line2D(
            [0], [0], color="w", marker=m, label=l, markerfacecolor=c, markersize=msize
        )
        # we use zipe to expand out markers etc.
        for m, c, l in zipe(markers, colors, labels)
    ]
    return custom_scats


def map_legend(raw_color_data, palette_data, marker, ax, is_legend_outside):
    """Creates and plots a legend to a figure."""
    names = unique_ordered(raw_color_data)
    cols = unique_ordered(palette_data)
    if isinstance(marker, str):
        markers = [marker] * len(names)
    else:
        marker_names = unique_ordered(marker)
        markers = tuple(it.islice(it.cycle(_marker_set()), 0, len(marker_names)))

    leg = legend_scatter(markers, cols, names)
    # does legend data/palette data have a title?
    if isinstance(raw_color_data, pd.Series):
        title = raw_color_data.name
    else:
        title = None
    # determine appropriate number of columns as divisible by 6
    rough_ncols = (len(names) // 6) + 1

    # add to an axes
    if is_legend_outside:
        ax.legend(leg, names, bbox_to_anchor=(1, 1), title=title, ncol=rough_ncols)
    else:
        ax.legend(leg, names, loc="best", title=title, ncol=rough_ncols)
