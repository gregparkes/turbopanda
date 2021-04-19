#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to random column selection in Metapanda."""

import numpy as np


def sample(self, p=1):
    """Fetches a random or multiple random columns as a subset of the dataset.

    Parameters
    ----------
    p : int
        The number of columns to return. Series if 1, DataFrame if >1

    Returns
    -------
    r : pd.Series/DataFrame
        Depending on size of `p`
    """
    return self.df_.iloc[:, np.random.choice(self.p, size=p, replace=False)].squeeze()
