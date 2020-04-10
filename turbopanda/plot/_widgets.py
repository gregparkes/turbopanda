#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creates custom legends."""

import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D

from turbopanda.utils import is_twotuple


def legend_line(group_to_colors, lw=3.):
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
        custom_lines = [Line2D([0], [0], color=colour, lw=lw) for label, colour in group_to_colors.items()]
    elif isinstance(group_to_colors, (list, tuple)):
        is_twotuple(group_to_colors)
        custom_lines = [Line2D([0], [0], color=colour, lw=lw) for label, colour in group_to_colors]
    elif isinstance(group_to_colors, (pd.Index, pd.Series)):
        custom_lines = [Line2D([0], [0], color=colour, lw=lw) for colour in group_to_colors.values]
    else:
        raise TypeError("type '{}' not recognized for `legend`".format(type(group_to_colors)))
    return custom_lines


def legend_scatter(markers, colors, labels, msize=15):
    """Given a mapping of markers, colors and labels, returns a scatterplot legend."""
    custom_scats = [Line2D([0], [0], color="w", marker=m, label=l,
                           markerfacecolor=c, markersize=msize)
                    for m, c, l in zip(markers, colors, labels)]
    return custom_scats
