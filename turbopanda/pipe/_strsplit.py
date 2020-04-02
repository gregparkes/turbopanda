#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to even more functions which can be directly fed into pandas.DataFrame.pipe.

These functions specifically handle string splitting.

Inspired from https://hackersandslackers.com/splitting-columns-with-pandas/
"""
from typing import List, Optional
import pandas as pd


__all__ = ('assign_split_col', 'assign_regex_col', 'split_list_like')


def assign_split_col(df: pd.DataFrame,
                     col: str,
                     name_list: List[str],
                     pat: Optional[str] = None) -> pd.DataFrame:
    """Splits on a column and assigns the new columns back into the DataFrame."""
    df_to_use = df.copy()
    split_col = df_to_use[col].str.split(pat, expand=True)

    return df_to_use.assign(
        **dict(
            zip(name_list, [split_col.iloc[:, x] for x in range(split_col.shape[1])])
        )
    )


def assign_regex_col(df: pd.DataFrame,
                     col: str,
                     name_list: List[str],
                     pat: Optional[str] = None) -> pd.DataFrame:
    """Extracts from a column and assigns the new columns back into the DataFrame."""
    df_to_use = df.copy()
    split_col = df_to_use[col].str.extract(pat, expand=True)

    return df.assign(
        **dict(
            zip(name_list, [split_col.iloc[:, x] for x in range(split_col.shape[1])])
        )
    )


def split_list_like(df: pd.DataFrame,
                    col: str,
                    new_col_prefix: str,
                    pat: Optional[str] = None):
    """Splits a list-like column painlessly."""
    df_to_use = df.copy()
    split_col = df_to_use[col].str.split(pat, expand=True)

    return df.assign(
        **{
            "{}_{}".format(new_col_prefix, x): split_col.iloc[:, x] for x in range(split_col.shape[1])
        }
    )
