#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the metadata preprocessing and creating in MetaPanda."""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Union, List, TypeVar, Dict, Callable
from ._types import SelectorType
from ._inspect import inspect
from turbopanda.utils import pairwise, intersect, union,\
    object_to_categorical, is_possible_category, instance_check, is_unique_id, t_numpy


__all__ = ('meta_map', 'update_meta', 'meta_split_category', 'sort_columns')


""" Basic column checking functions. """


def true_type(ser: pd.Series) -> TypeVar:
    """
    Given a pandas.Series, determine it's true datatype if it has missing values.
    """
    # return 'reduced' Series if we're missing data and dtype is not an object, else just return the default dtype

    return pd.to_numeric(ser.dropna(), errors="ignore", downcast="unsigned").dtype \
        if ((ser.dtype in t_numpy()) and (ser.count() > 0)) else ser.dtype


def is_mixed_type(ser: pd.Series) -> bool:
    """Determines whether the column has mixed types in it."""
    return ser.map(lambda x: type(x)).nunique() > 1 if ser.dtype == object else False


""" Prepares dictionary of current meta columns. """


def default_columns() -> Dict[str, Callable]:
    """The default metadata columns provided."""
    return {"true_type": true_type,
            "is_mixed_type": is_mixed_type,
            "is_unique_id": is_unique_id
            }


""" Constructs a basic meta dataset. """


def basic_construct(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a basic meta file."""
    _meta = pd.DataFrame({}, index=df.columns)
    _meta.index.name = "colnames"
    return _meta


""" Iterates through and converts columns to type category. """


def categorize_meta(meta: pd.DataFrame):
    """
    Go through the meta_ attribute and convert possible objects to type category.

    Modifies meta inplace
    """
    for column in meta.columns:
        if is_possible_category(meta[column]):
            meta[column] = object_to_categorical(meta[column])


def dummy_categorical(cat: pd.Series) -> pd.DataFrame:
    """Given pd.Series of type 'category', return boolean dummies as matrix."""
    instance_check(cat, pd.Series)
    if cat.dtype == 'category':
        return pd.get_dummies(cat).add_prefix("is_").astype(np.bool)
    else:
        raise TypeError("'cat' Series is {}, not of type 'category'".format(cat.dtype))


def _add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """ Constructs a pd.DataFrame from the raw data. Returns meta"""
    # step 1. construct a DataFrame based on the column names as an index.
    _func_mapping = default_columns()
    _agg = df.aggregate(list(_func_mapping.values())).T
    # cast bool-like columns to bool
    _agg['is_mixed_type'] = _agg['is_mixed_type'].astype(bool)
    _agg['is_unique_id'] = _agg['is_unique_id'].astype(bool)
    # return
    return _agg


""" Local metapanda functions. """


def _create_new_metamap(df, meta, selectors, mapper, name, meta_set):
    # for each selector, get the group view.
    if isinstance(meta_set, (list, tuple)):
        cnames = [inspect(df, meta, selectors, sel, mode='view') for sel in meta_set]
    else:
        raise TypeError("'selectors' must be of type {list, tuple}")

    # calculate the pairwise intersection between all the cnames
    igrid = union(*pairwise(intersect, cnames))

    if len(igrid) == 0:
        new_grid = pd.concat([pd.Series(n, index=val) for n, val in zip(meta_set, cnames)], sort=False, axis=0)
        new_grid.name = name
    else:
        raise ValueError("shared terms: {} discovered for meta_map.".format(igrid))
    # merge into meta
    cat = object_to_categorical(new_grid, meta_set)
    cat.name = name
    # APPARENTLY CONCAT doesn't work here? for some dumb reason.
    meta = meta.join(cat)
    # store meta_map for future reference.
    mapper[name] = meta_set


def _redefine_metamaps(df, meta, mapper, selectors):
    if len(mapper) > 0:
        for k, v in mapper.items():
            _create_new_metamap(df, meta, selectors, mapper, k, v)


def _reset_meta(df, meta, mapper, selectors) -> pd.DataFrame:
    """Returns a meta dataset. """
    # add in metadata rows.
    _meta = _add_metadata(df)
    # if we have mapper elements, add these in
    _redefine_metamaps(df, meta, mapper, selectors)
    return _meta


def meta_map(self, name: str,
             selectors: List[SelectorType]) -> "MetaPanda":
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
        Each contains either types, meta column names, column names or regex-compliant strings

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
    _create_new_metamap(self.df_, self.meta_,
                        self.selectors_, self.mapper_, name, selectors)
    return self


def update_meta(self) -> "MetaPanda":
    """Forces an update to the metadata.

    This involves a full `meta_` reset, so columns present may be lost.

    .. warning:: This is experimental and may disappear or change in future updates.

    Returns
    -------
    self
    """
    # should include a call to define the meta maps.
    self._meta = _reset_meta(self.df_, self.meta_, self.mapper_, self.selectors_)
    return self


def meta_split_category(self, cat: str) -> "MetaPanda":
    """Splits category into k boolean columns in `meta_` to use for selection.

    This enables a categorical column to contain multiple boolean selectors for
    downstream use.

    The categorical column is kept and not removed. New columns are concatenated on.

    Parameters
    ----------
    cat : str
        The name of the `meta_` column to split on.

    Raises
    ------
    ValueError
        If `cat` column is not found in `meta_`.
        If resulting columns already exist in `meta_`.

    Returns
    -------
    self
    """
    if cat in self.meta_:
        # expand and add to meta.
        try:
            self._meta = pd.concat([
                self.meta_, dummy_categorical(self.meta_[cat])
                # integrity must be verified to make sure these columns do not already exist.
            ], sort=False, axis=1, join="inner", copy=True, verify_integrity=True)
        except ValueError:
            warnings.warn("in `meta_split_category`: integrity of meta_ column challenged, no split has occurred.",
                          UserWarning)
        return self
    else:
        raise ValueError("cat column '{}' not found in `meta_`.".format(cat))


def sort_columns(self,
                 by: Union[str, List[str]] = "colnames",
                 ascending: Union[bool, List[bool]] = True) -> "MetaPanda":
    """Sorts `df_` using vast selection criteria.

    Parameters
    -------
    by : str, list of str, optional
        Sorts columns based on information in `meta_`, or by alphabet, or by index.
        Accepts {'colnames'} as additional options. 'colnames' is `index`
    ascending : bool, list of bool, optional
        Sort ascending vs descending.
        If list, specify multiple ascending/descending combinations.

    Raises
    ------
    ValueException
        If the length of `by` does not equal the length of `ascending`, in list instance.
    TypeException
        If `by` or `ascending` is not of type {list}

    Returns
    -------
    self
    """
    if isinstance(by, str):
        by = [by]
    if isinstance(by, list) and isinstance(ascending, (bool, list)):
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        elif len(by) != len(ascending):
            raise ValueError(
                "the length of 'by' {} must equal the length of 'ascending' {}".format(len(by), len(ascending)))
        if all([(col in self.meta_) or (col == "colnames") for col in by]):
            self._meta = self.meta_.sort_values(by=by, axis=0, ascending=ascending)
            self._df = self._df.reindex(self.meta_.index, axis=1)
    else:
        raise TypeError("'by' or 'ascending' is not of type {list}")
    return self