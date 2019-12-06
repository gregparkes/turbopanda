#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:10:17 2019

@author: gparkes

Handling the construction and use of metadata associated with MetaPanda
"""

import numpy as np
import pandas as pd
from .utils import is_missing_values, is_unique_id, is_potential_id, is_potential_stacker, nunique


def reduce_numeric_series(ser):
    """
    Given a pandas.Series, determine it's true datatype if it has missing values.
    """
    # return 'reduced' Series if we're missing data and dtype is not an object, else just return the default dtype
    return pd.to_numeric(ser.dropna(), errors="ignore", downcast="unsigned").dtype if (ser.dtype != object) else ser.dtype


def basic_construct(df):
    _meta = pd.DataFrame({}, index=df.columns)
    _meta.index.name = "colnames"
    return _meta


def add_metadata(df, curr_meta):
    """ Constructs a pd.DataFrame from the raw data. Returns meta"""
    # step 1. construct a DataFrame based on the column names as an index.

    expected_types = [reduce_numeric_series(df[c]) for c in df]
    is_uniq = [is_unique_id(df[c]) for c in df]
    is_id = [is_potential_id(df[c]) for c in df]
    is_stacked = [is_potential_stacker(df[c]) for c in df]
    is_missing = [is_missing_values(df[c]) for c in df]
    nun = [nunique(df[c]) for c in df]

    loc_mapping = {
        "e_types": expected_types,
        "is_unique": is_uniq,
        "is_potential_id": is_id,
        "is_potential_stacker":is_stacked,
        "is_missing": is_missing,
        "n_uniques": nun,
    }

    for key, values in loc_mapping.items():
        curr_meta[key] = values
