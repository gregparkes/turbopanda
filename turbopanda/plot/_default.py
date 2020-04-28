#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for different defaults to use in conjunction with Matplotlib."""

import matplotlib as mpl

from turbopanda._deprecator import unimplemented


def set_style(style_name="paper"):
    """Generates a dictionary of styles to incorporate into plots.

    Choose from {'paper'}
    """
    _styles = {
        "paper": {'axes.labelsize': 8, 'text.fontsize': 8, 'legend.fontsize': 10,
                  'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.usetex': False,
                  'figure.figsize': [4.5, 4.5]}
    }

    _selected_style = _styles[style_name]
    mpl.rcParams.update(_selected_style)
