#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overloaded head() function for pandas."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from typing import Optional, Tuple, Union


def expand(self, column: str, sep: str = ",") -> "MetaPanda":
    """Expand out a 'stacked' id column to a longer-form DataFrame.

    Expands out a 'stacked' id column to a longer-form DataFrame, and re-merging
    the data back in.

    Parameters
    ----------
    column : str
        The name of the column to expand, must be of datatype [object]
    sep : str, optional
        The separating string to use.

    Raises
    ------
    ValueError
        If `column` not found in `df_` or `meta_`, or `column` is not stackable

    Returns
    -------
    self

    See Also
    --------
    shrink : Expands out a 'unstacked' id column to a shorter-form DataFrame.
    """
    if column not in self.df_.columns:
        raise ValueError("column '{}' not found in df".format(column))
    if not self.meta_.loc[column, "is_potential_stacker"]:
        raise ValueError("column '{}' not found to be stackable".format(column))

    self._df = pd.merge(
        # expand out id column
        self.df_[column].str.strip().str.split(sep).explode(),
        self.df_.dropna(subset=[column]).drop(column, axis=1),
        left_index=True, right_index=True
    )
    self.df_.columns.name = "colnames"
    return self


def shrink(self, column: str, sep: str = ",") -> "MetaPanda":
    """Expand out a 'unstacked' id column to a shorter-form DataFrame.

    Shrinks down a 'duplicated' id column to a shorter-form dataframe, and re-merging
    the data back in.

    Parameters
    -------
    column : str
        The name of the duplicated column to shrink, must be of datatype [object]
    sep : str, optional
        The separating string to add.

    Raises
    ------
    ValueError
        If `column` not found in `df_` or `meta_`, or `column` is not shrinkable

    Returns
    -------
    self

    See Also
    --------
    expand : Expands out a 'stacked' id column to a longer-form DataFrame.
    the data back in.
    """
    if (column not in self.df_.columns) and (column != self.df_.index.name):
        raise ValueError("column '{}' not found in df".format(column))

    # no changes made to columns, use hidden df
    self._df = pd.merge(
        # shrink down id column
        self.df_.groupby("counter")[column].apply(lambda x: x.str.cat(sep=sep)),
        self.df_.reset_index().drop_duplicates("counter").set_index("counter").drop(column, axis=1),
        left_index=True, right_index=True
    )
    self.df_.columns.name = "colnames"
    return self


def split_categories(self,
                     column: str,
                     sep: str = ",",
                     renames: Optional[Tuple[str, ...]] = None) -> "MetaPanda":
    """Split a column into N categorical variables to be associated with df_.

    Parameters
    ----------
    column : str
        The name of the column to split, must be of datatype [object], and contain values sep inside
    sep : str, optional
        The separating string to add.
    renames : None or list of str, optional
        If list of str, must be the same dimension as expanded columns

    Raises
    ------
    ValueError
        `column` not found in `df_` column names

    Returns
    -------
    self
    """
    if column not in self.df_.columns:
        raise ValueError("column '{}' not found in df".format(column))

    exp = self.df_[column].str.strip().str.split(sep, expand=True)
    # calculate column names
    if renames is None:
        cnames = ["cat%d" % (i + 1) for i in range(exp.shape[1])]
    else:
        cnames = renames if len(renames) == exp.shape[1] else ["cat%d" % (i + 1) for i in range(exp.shape[1])]

    self._df = self.df_.join(
        exp.rename(columns=dict(zip(range(exp.shape[1]), cnames)))
    )
    self.df_.columns.name = "colnames"
    return self


def sort_columns(self,
                 by: Union[str, Tuple[str, ...]] = "colnames",
                 ascending: Union[bool, Tuple[bool, ...]] = True) -> "MetaPanda":
    """Sorts `df_` using vast selection criteria.

    Parameters
    -------
    by : str, tuple of str, optional
        Sorts columns based on information in `meta_`, or by alphabet, or by index.
        Accepts {'colnames'} as additional options. 'colnames' is `index`
    ascending : bool, tuple of bool, optional
        Sort ascending vs descending.
        If list/tuple, specify multiple ascending/descending combinations.

    Raises
    ------
    ValueException
        If the length of `by` does not equal the length of `ascending`, in list instance.
    TypeException
        If `by` or `ascending` is not of type {list, tuple}

    Returns
    -------
    self
    """
    if isinstance(by, str):
        by = [by]
    if isinstance(by, tuple) and isinstance(ascending, (bool, tuple)):
        if len(by) != len(ascending):
            raise ValueError(
                "the length of 'by' {} must equal the length of 'ascending' {}".format(len(by), len(ascending)))
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        if all([(col in self.meta_) or (col == "colnames") for col in by]):
            self._meta = self.meta_.sort_values(by=by, axis=0, ascending=ascending)
            self._df = self._df.reindex(self.meta_.index, axis=1)
    else:
        raise TypeError("'by' or 'ascending' is not of type {list, tuple}")
    return self
