#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty histograms in primitive matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    return stats.scoreatpercentile(np.asarray(a), 75) - stats.scoreatpercentile(np.asarray(a), 25)


def freedman_diaconis_bins(a: np.ndarray) -> int:
    """
    Calculate number of hist bins using Freedman-Diaconis rule.

    Taken from https://github.com/mwaskom/seaborn/blob/master/seaborn/distributions.py
    """
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * _iqr(a) / (a.shape[0] ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((np.nanmax(a) - np.nanmin(a)) / h))


def histogram(X, bins=None, density=True, kde=True, ax=None, x_label=None, title=None):
    """Draws pretty histograms using `X`.

    Parameters
    ----------
    X : np.ndarray/pd.Series (1d)
        The data column to draw.
    bins : int, optional
        If None, uses optimal algorithm to find best bin count
    density : bool
        If True, uses density approximation
    kde : bool
        If True, uses a kernel density approximation, and uses `normed`
    ax : matplotlib.ax object, optional
        If None, creates one.
    x_label : str, optional
        If None, uses `x-axis`.
    title : str, optional
        If None, uses `Default Title`

    Returns
    -------
    None
    """
    if bins is None:
        bins = min(freedman_diaconis_bins(X), 50)
    if kde:
        density = True
    # plot histogram
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if x_label is None:
        x_label = "x-axis"
    if title is None:
        title = "Default Title"

    _ = ax.hist(X, bins=bins, density=density, color='g')
    ax.set_xlabel(x_label)
    ax.set_title(title)

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")

    if kde:
        X_std = X.std()
        x_kde = np.linspace(X.min()-X_std, X.max()+X_std, 200)
        # fit params to normal distribution
        params = stats.norm.fit(X)
        # generate y kde
        y_kde = stats.norm.pdf(x_kde, *params)
        # plot
        ax.plot(x_kde, y_kde, "-", color='r')

    plt.show()
