#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles whether a dataset is multivariate-normal and returns those datas."""

import pandas as pd
from scipy.stats import probplot


def is_mvn(df, r_thresh=0.95):
    """Determines whether columns in df are normally-distributed.

    Parameters
    ----------
    df : dataset (pd.DataFrame)
        The full dataset
    r_thresh : float
        The threshold correlation to choose.

    Returns
    -------
    ind_dims : pd.Index
        The columns that are normally distributed
    """
    pppbs = [probplot(df[x])[-1] for x in df.columns]
    # convert to pdf
    pppbs_df = pd.DataFrame(pppbs, index=df.columns, columns=('slope','intercept','r'))
    # find samples that have r > r_thresh
    ind_dims = pppbs_df[pppbs_df['r'] > r_thresh].index
    return ind_dims
