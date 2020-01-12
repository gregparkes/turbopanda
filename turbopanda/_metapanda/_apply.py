#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renaming the column names, and variants thereof."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd


def _apply_function(df: pd.DataFrame,
                    fn: str,
                    *fargs,
                    **fkwargs):
    """Executes a change to pandas.DataFrame."""
    if hasattr(df, fn):
        df = getattr(df, fn)(*fargs, **fkwargs)
        return df
    # if we start with groupby, then we are coupling groupby with an aggregation.
    elif fn.startswith("groupby__"):
        _, fn2 = fn.split("__", 1)
        _grouped = getattr(df, "groupby")(*fargs, **fkwargs)
        if hasattr(_grouped, fn2):
            df = _grouped.agg(fn2)
            return df
        else:
            raise ValueError(
                "function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn2, dir(_grouped)))
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn, dir(df)))


def _apply_index_function(df: pd.DataFrame,
                          fn: str,
                          *fargs,
                          **fkwargs):
    if hasattr(df.index, fn):
        df.index = getattr(df.index, fn)(*fargs, **fkwargs)
    elif hasattr(df.index.str, fn):
        df.index = getattr(df.index.str, fn)(*fargs, **fkwargs)
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.index.[str.]* API".format(fn))


def _apply_column_function(df: pd.DataFrame,
                           meta: pd.DataFrame,
                           fn: str,
                           *fargs,
                           **fkwargs):
    if hasattr(df.columns, fn):
        df.columns = getattr(df.columns, fn)(*fargs, **fkwargs)
        df.index = getattr(meta.index, fn)(*fargs, **fkwargs)
    elif hasattr(df.columns.str, fn):
        df.columns = getattr(df.columns.str, fn)(*fargs, **fkwargs)
        df.index = getattr(meta.index.str, fn)(*fargs, **fkwargs)
    else:
        raise ValueError("function '{}' not recognized in pandas.DataFrame.columns.[str.]* API".format(fn))


def apply(self,
          f_name: str,
          *f_args,
          **f_kwargs) -> "MetaPanda":
    """Apply a `pd.DataFrame` function to `df_`.

    e.g mdf.apply("groupby", ["counter","refseq_id"], as_index=False)
        applies self.df_.groupby() to data and return value is stored in df_
        assumes pandas.DataFrame is returned.

    Parameters
    -------
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
    self._df = _apply_function(self.df_, f_name, *f_args, **f_kwargs)
    return self


def apply_columns(self,
                  f_name: str,
                  *f_args,
                  **f_kwargs) -> "MetaPanda":
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
    _apply_column_function(self.df_, self.meta_, f_name, *f_args, **f_kwargs)
    return self


def apply_index(self,
                f_name: str,
                *f_args,
                **f_kwargs) -> "MetaPanda":
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
    _apply_index_function(self.df_, f_name, *f_args, **f_kwargs)
    return self