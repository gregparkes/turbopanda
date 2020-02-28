#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the application of pandas operations in Metapanda."""

import pandas as pd
import warnings
from turbopanda.utils import instance_check, is_column_string


__all__ = ('apply', 'apply_index', 'apply_columns', '_apply_function', '_apply_index_function', '_apply_column_function')


def _apply_function(self, fn: str, *fargs, **fkwargs):
    if hasattr(self.df_, fn):
        self._df = getattr(self.df_, fn)(*fargs, **fkwargs)
        return self
    # if we start with groupby, then we are coupling groupby with an aggregation.
    elif fn.startswith("groupby__"):
        _, fn2 = fn.split("__", 1)
        _grouped = getattr(self.df_, "groupby")(*fargs, **fkwargs)
        if hasattr(_grouped, fn2):
            self._df = _grouped.agg(fn2)
            return self
        else:
            raise ValueError(
                "function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn2, dir(_grouped)))
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn, dir(self.df_)))


def _apply_index_function(self, fn: str, *fargs, **fkwargs):
    if hasattr(self.df_.index, fn):
        self.df_.index = getattr(self.df_.index, fn)(*fargs, **fkwargs)
        return self
    elif hasattr(self.df_.index.str, fn):
        if is_column_string(self.df_.index):
            self.df_.index = getattr(self.df_.index.str, fn)(*fargs, **fkwargs)
        else:
            warnings.warn(
                "operation pandas.Index.str.'{}' cannot operate on index because they are not of type str.".format(
                    fn),
                PendingDeprecationWarning
            )
        return self
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.index.[str.]* API".format(fn))


def _apply_column_function(self, fn: str, *fargs, **fkwargs):
    if hasattr(self.df_.columns, fn):
        self.df_.columns = getattr(self.df_.columns, fn)(*fargs, **fkwargs)
        self.meta_.index = getattr(self.meta_.index, fn)(*fargs, **fkwargs)
        return self
    elif hasattr(self.df_.columns.str, fn):
        if is_column_string(self.df_.columns) and is_column_string(self.meta_.index):
            self.df_.columns = getattr(self.df_.columns.str, fn)(*fargs, **fkwargs)
            self.meta_.index = getattr(self.meta_.index.str, fn)(*fargs, **fkwargs)
        else:
            warnings.warn(
                "operation pandas.Index.str.'{}' invalid because column/index is not of type str.".format(
                    fn),
                PendingDeprecationWarning
            )
        return self
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.columns.[str.]* API".format(fn))


def apply(self, f_name: str, *f_args, **f_kwargs) -> "MetaPanda":
    """Apply a `pd.DataFrame` function to `df_`.

    e.g mdf.apply("groupby", ["counter","refseq_id"], as_index=False)
        applies self.df_.groupby() to data and return value is stored in df_
        assumes pandas.DataFrame is returned.

    Parameters
    ----------
    f_name : str
        The name of the function
    f_args : list/tuple, optional
        Arguments to pass to the function
    f_kwargs : dict, optional
        Keyword arguments to pass to the function
    Returns
    -------
    self
    """
    instance_check(f_name, str)
    self._apply_function(f_name, *f_args, **f_kwargs)
    return self


def apply_columns(self, f_name: str, *f_args, **f_kwargs) -> "MetaPanda":
    """Apply a `pd.Index` function to `df_.columns`.

    The result is then returned to the columns attribute, so it should only accept transform-like operations.

    Thus to apply `strip` to all column names:

    >>> import turbopanda as turb
    >>> mdf = turb.MetaPanda()
    >>> mdf.apply_columns("strip")

    Parameters
    -------
    f_name : str
        The name of the function. This can be in the .str accessor attribute also.
    f_args : list/tuple, optional
        Arguments to pass to the function
    f_kwargs : dict, optional
        Keyword arguments to pass to the function

    Returns
    -------
    self
    """
    self._apply_column_function(f_name, *f_args, **f_kwargs)
    return self


def apply_index(self, f_name: str, *f_args, **f_kwargs) -> "MetaPanda":
    """Apply a `pd.Index` function to `df_.index`.

    The result is then returned to the index attribute, so it should only accept transform-like operations.

    Thus to apply `strip` to all index names:

    >>> import turbopanda as turb
    >>> mdf = turb.MetaPanda()
    >>> mdf.apply_columns("strip")

    Parameters
    -------
    f_name : str
        The name of the function. This can be in the .str accessor attribute also.
    f_args : list/tuple, optional
        Arguments to pass to the function
    f_kwargs : dict, optional
        Keyword arguments to pass to the function

    Returns
    -------
    self
    """
    self._apply_index_function(f_name, *f_args, **f_kwargs)
    return self
