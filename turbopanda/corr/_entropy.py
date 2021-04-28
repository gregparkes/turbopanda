#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles some mutual information calculations."""

# future imports
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import entropy as stat_entropy

from turbopanda.stats._kde import freedman_diaconis_bins
from turbopanda.utils import instance_check, as_flattened_numpy


def _entropy(X, Y=None, bins=20):
    # X, Y are np.ndarrays
    if Y is None:
        hist, b_edge = np.histogram(X, bins=bins, density=True)
        return stat_entropy(hist, base=2)
    else:
        hist, b_edge_x, b_edge_y = np.histogram2d(X, Y, bins=bins, density=True)
        return stat_entropy(hist.flatten(), base=2)


def continuous_mutual_info(
        data: pd.DataFrame,
        x: str,
        y: str,
        bins: Optional[int] = None
) -> float:
    """Determines mutual information given random variables.

    .. math:: I(X;Y) = H(X) + H(Y) - H(X,Y)

    where :math:`H(X,Y)` is the joint probability, and :math:`H(X)` is the marginal distribution
        of X.

    Entropy is estimated using Shannon entropy method.
        Provides for conditional calculations.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to perform the operation upon.
    x : str
        X column continuous to select
    y : str
        Y column continuous to select
    bins : int, optional
        The number of bins, if None uses freedman_diaconis algorithm.

    Returns
    -------
    MI : float
        I(X; Y)
    """
    # create subset by removing missing values.
    DF = data[[x, y]].dropna()
    _X = DF[x].values
    _Y = DF[y].values

    # generate bins
    if bins is None:
        bins = min(freedman_diaconis_bins(_X), 50)

    H_X = _entropy(_X, bins=bins)
    H_Y = _entropy(_Y, bins=bins)
    H_XY = _entropy(_X, _Y, bins=bins)
    return H_X + H_Y - H_XY
