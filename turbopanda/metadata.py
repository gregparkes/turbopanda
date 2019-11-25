#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:10:17 2019

@author: gparkes

Handling the construction and use of metadata associated with MetaPanda
"""

import pandas as pd
from .utils import not_column_float, is_column_object


def is_unique_ID(ser):
    # a definite algorithm for determining a unique column ID
    return ser.is_unique if not_column_float(ser) else False


def is_potential_ID(ser, thresh=0.5):
    return (ser.unique().shape[0] / ser.shape[0]) > thresh if not_column_float(ser) else False


def is_potential_stacker(ser, regex=";|\t|,|\|", thresh=0.1):
    return ser.str.contains(regex).sum() > thresh if is_column_object(ser) else False


def construct_meta(df):
    """ Constructs a pd.DataFrame from the raw data. Returns meta"""
    # step 1. construct a dataframe based on the column names as an index.
    colnames = df.columns
    is_uniq = [is_unique_ID(df[c]) for c in df]
    is_id = [is_potential_ID(df[c]) for c in df]
    is_stacked = [is_potential_stacker(df[c]) for c in df]

    # FINAL - return all as a meta_ option.
    _meta = pd.DataFrame({
        "mtypes": df.dtypes.values,
        "is_unique": is_uniq,
        "potential_id": is_id,
        "potential_stacker":is_stacked
    }, index=colnames)
    _meta.index.name = "colnames"
    return _meta
