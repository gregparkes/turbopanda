#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty 2D KDEs in primitive matplotlib."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from pandas import Series

from turbopanda._deprecator import unimplemented
from turbopanda.stats import density
from turbopanda.utils import remove_na, instance_check, arrays_equal_size


@unimplemented
def kde2d(X: Union[np.ndarray, Series, List, Tuple],
          Y: Union[np.ndarray, Series, List, Tuple],
          c: str = 'red',
          ax: mpl.axes.Axes = None,
          fill: bool = False,
          with_scatter: bool = False,
          **contour_kwargs):
    """TODO: Generates a 2D KDE using contours."""
    instance_check((X, Y), (list, tuple, np.ndarray, Series))
    instance_check(c, str)
    instance_check((fill, with_scatter), bool)
    instance_check(ax, matplotlib.axes.Axes)
    arrays_equal_size(X, Y)

    # calculate density
    _X, _Y = remove_na(np.asarray(X), np.asarray(Y), paired=True)

    H = density(_X, _Y)
    offx = np.abs(_X.max() - _X.min()) / 15.
    offy = np.abs(_Y.max() - _Y.min()) / 15.
    _alpha = .5 if with_scatter else 1.

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if fill:
        ax.contourf(H, extent=(_X.min() - offx, _X.max() + offx, _Y.min() - offy, _Y.max() + offy), color=c,
                    alpha=_alpha)
    else:
        cset = ax.contour(H, extent=(_X.min() - offx, _X.max() + offx, _Y.min() - offy, _Y.max() + offy), color=c,
                          **contour_kwargs)
        ax.clabel(cset, inline=1, fontsize=10)

    if with_scatter:
        ax.scatter(_X, _Y, c=c, alpha=_alpha)

    return ax
