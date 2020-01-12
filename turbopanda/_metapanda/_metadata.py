#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handling metadata functions."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from typing import Tuple
import sys
sys.path.append("../")
from .custypes import SelectorType
from .utils import object_to_categorical
from .analyze import intersection_grid


def _reduce_data_type(ser: pd.Series):
    """
    Given a pandas.Series, determine it's true datatype if it has missing values.
    """
    # return 'reduced' Series if we're missing data and dtype is not an object, else just return the default dtype

    return pd.to_numeric(ser.dropna(), errors="ignore", downcast="unsigned").dtype \
        if ((ser.dtype != object) and (ser.dropna().shape[0] > 0)) else ser.dtype


def _is_mixed_type(ser: pd.Series) -> bool:
    return ser.apply(lambda x: type(x)).unique().shape[0] > 1


def _is_unique_id(ser: pd.Series) -> bool:
    """Determine whether ser is unique."""
    return ser.is_unique if is_column_int(ser) else False


def _is_potential_id(ser: pd.Series, thresh: float = 0.5) -> bool:
    """Determine whether ser is a potential ID column."""
    return (ser.unique().shape[0] / ser.shape[0]) > thresh if is_column_int(ser) else False


def _is_potential_stacker(ser: pd.Series, regex: str = ";|\t|,|", thresh: float = 0.1) -> bool:
    """Determine whether ser is a stacker-like column."""
    return ser.dropna().str.contains(regex).sum() > thresh if (ser.dtype == object) else False


def _is_missing_values(ser: pd.Series) -> bool:
    """Determine whether any missing values are present."""
    return ser.count() < ser.shape[0]


def _nunique(ser: pd.Series) -> int:
    """Convert ser to be nunique."""
    return ser.nunique() if not_column_float(ser) else -1


def _meta_columns_default() -> Tuple[str, ...]:
    """The default metadata columns provided."""
    return ("e_types", "is_unique", "is_potential_id", "is_potential_stacker",
            "is_missing", "n_uniques")


def _categorize_meta(meta: pd.DataFrame):
    """
    Go through the meta_ attribute and convert possible objects to type category.

    Modifies meta inplace
    """
    for column in meta.columns:
        if is_possible_category(meta[column]):
            meta[column] = object_to_categorical(meta[column])


def _basic_construct(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a basic meta file."""
    _meta = pd.DataFrame({}, index=df.columns)
    _meta.index.name = "colnames"
    return _meta


def _add_metadata(df: pd.DataFrame, curr_meta: pd.DataFrame):
    """ Constructs a pd.DataFrame from the raw data. Returns meta"""
    # step 1. construct a DataFrame based on the column names as an index.
    loc_mapping = {
        "e_types": [_reduce_data_type(df[c]) for c in df],
        "is_mixed_type": [_is_mixed_type(df[c]) for c in df],
        "is_unique": [_is_unique_id(df[c]) for c in df],
        "is_potential_id": [_is_potential_id(df[c]) for c in df],
        "is_potential_stacker": [_is_potential_stacker(df[c]) for c in df],
        "is_missing": [_is_missing_values(df[c]) for c in df],
        "n_uniques": [_nunique(df[c]) for c in df],
    }
    # add to the metadata.
    for key, values in loc_mapping.items():
        curr_meta[key] = values


def update_meta(self):
    if len(self.mapper_) > 0:
        for k, v in self.mapper_.items():
            self.meta_map(k, v)


def _reset_meta(self):
    self._meta = _basic_construct(self._df)
    # add in metadata rows.
    _add_metadata(self._df, self._meta)
    # if we have mapper elements, add these in
    if len(self.mapper_) > 0:
        self.update_meta()


def meta_map(self,
             name: str,
             selectors: Tuple[SelectorType, ...]) -> "MetaPanda":
    """Map a group of selectors with an identifier, in `mapper_`.

    Maps a group of selectors into a column in the meta-information
    describing some groupings/associations between features. For example,
    your data may come from multiple sources and you wish to
    identify this within your data.

    Parameters
    --------
    name : str
        The name of this overall grouping
    selectors : list/tuple of (str, or tuple args)
        Each contains either custypes.py, meta column names, column names or regex-compliant strings

    Raises
    ------
    TypeException
        selectors must be of type {list, tuple}
    ValueException
        If terms overlap within selector groups

    Returns
    -------
    self

    See Also
    --------
    cache : Adds a cache element to `selectors_`.
    cache_k : Adds k cache elements to `selectors_`.
    """
    # for each selector, get the group view.
    if isinstance(selectors, (list, tuple)):
        cnames = [self.view(sel) for sel in selectors]
    else:
        raise TypeError("'selectors' must be of type {list, tuple}")

    igrid = intersection_grid(cnames)
    if igrid.shape[0] == 0:
        new_grid = pd.concat([pd.Series(n, index=val) for n, val in zip(selectors, cnames)], sort=False, axis=0)
        new_grid.name = name
    else:
        raise ValueError("shared terms: {} discovered for meta_map.".format(igrid))
    # merge into meta
    self.meta_[name] = object_to_categorical(new_grid, selectors)
    # store meta_map for future reference.
    self.mapper_[name] = selectors
    return self