#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:18:39 2019

@author: gparkes
"""

__all__ = ["low_sample", "low_variance"]


def low_sample(df, thresh=0.5):
    """
    Given a dataframe, determine which columns to keep
    as they have a proportion of non-NA values above this threshold.

    Parameters
    -------
    df : pd.DataFrame
        The dataset to perform on
    thresh : float [0..1]
        A threshold to determine at what point to keep a column

    Returns
    -------
    ser : pd.Series
        A boolean series
    """
    return df.count().div(df.shape[0]).gt(thresh)


def low_variance(df, thresh=1e-9):
    """
    Given a dataframe, determine which columns to keep
    as they have a proportion of extremely low variance (nothing interesting).
    Only works on float columns.

    Parameters
    -------
    df : pd.DataFrame
        The dataset to perform on
    thresh : float [0..1]
        A threshold to determine at what point to keep a column

    Returns
    -------
    ser : pd.Series
        A boolean series
    """
    return df.var().gt(thresh)
