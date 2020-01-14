#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handling the construction and use of metadata associated with MetaPanda."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import pandas as pd
from typing import Tuple, Callable, Dict

# locals
from .utils import is_missing, is_unique_id, is_possible_category, object_to_categorical, t_numpy

__all__ = ("default_columns", "basic_construct", "categorize_meta", "add_metadata")


def true_type(ser: pd.Series):
    """
    Given a pandas.Series, determine it's true datatype if it has missing values.
    """
    # return 'reduced' Series if we're missing data and dtype is not an object, else just return the default dtype

    return pd.to_numeric(ser.dropna(), errors="ignore", downcast="unsigned").dtype \
        if ((ser.dtype in t_numpy()) and (ser.count() > 0)) else ser.dtype


def is_mixed_type(ser: pd.Series) -> bool:
    """Determines whether the column has mixed types in it."""
    return ser.map(lambda x: type(x)).nunique() > 1 if ser.dtype == object else False


def default_columns() -> Dict[str, Callable]:
    """The default metadata columns provided."""
    return {"true_type": true_type,
            "is_mixed_type": is_mixed_type,
            "is_unique_id": is_unique_id
    }


def basic_construct(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a basic meta file."""
    _meta = pd.DataFrame({}, index=df.columns)
    _meta.index.name = "colnames"
    return _meta


def categorize_meta(meta: pd.DataFrame):
    """
    Go through the meta_ attribute and convert possible objects to type category.

    Modifies meta inplace
    """
    for column in meta.columns:
        if is_possible_category(meta[column]):
            meta[column] = object_to_categorical(meta[column])


def add_metadata(df: pd.DataFrame, columns=None):
    """ Constructs a pd.DataFrame from the raw data. Returns meta"""
    # step 1. construct a DataFrame based on the column names as an index.
    _func_mapping = default_columns()
    return df.aggregate(list(_func_mapping.values())).T
