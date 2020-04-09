#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty 2D KDEs in primitive matplotlib."""

import numpy as np
import matplotlib.pyplot as plt

from turbopanda.stats import density
from turbopanda.utils import remove_na


def kde2d(X, Y, c='red', ax=None, fill=False, with_scatter=False, **contour_kwargs):
    """TODO: Generates a 2D KDE using contours."""
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
