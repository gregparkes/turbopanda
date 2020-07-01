#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling string operations on pandas.DataFrames and more.
We collapse down all of the different routes into a catch-all `pattern` function which will discern the pattern
from a bunch of different options.
"""
import re
import pandas as pd
from pandas import Index
from typing import Tuple, Union, Iterable, List

from turbopanda.utils import set_like, union, intersect, absdifference, \
    instance_check


def _strpattern(pat, K):
    """Determines if pattern `pat` exists in list of str `K`."""
    # compile pattern - improves performance
    _p = re.compile(pat)
    # iterate over and return
    return set_like([s for s in K if re.search(_p, s)])


def _patcolumnmatch(pat, df):
    """Use regex to match to column names in a pandas.DataFrame."""
    _p = re.compile(pat)
    c_fetch = [s for s in df.columns if re.search(_p, s)]
    if len(c_fetch) > 0:
        return Index(c_fetch, dtype=object, name=df.columns.name,
                     tupleize_cols=False)
    else:
        return Index([], name=df.columns.name)


def _choose_regex(pat, K):
    if isinstance(K, pd.DataFrame):
        return _patcolumnmatch(pat, K)
    elif isinstance(K, pd.Series):
        return _strpattern(pat, K.values)
    else:
        return _strpattern(pat, K)


def _foreach_flexterm(term, K):
    if term.startswith("~"):
        s = _choose_regex(term[1:], K)
        res = absdifference(K, s)
    else:
        res = _choose_regex(term, K)
    return res


def pattern(pat: str, K, extended_regex: bool = True):
    """Determines if pattern `pat` exists in K.

    Parameters
    ----------
    pat : str
        Regex-compliant pattern. Also supports *flexible* regex with NOT operators in the case
        of pandas.DataFrame columns, etc.
    K : list, tuple, pd.Series, pd.Index, pd.DataFrame
        The full list to select a pattern from. Accepts pandas. inputs, and uses the .columns attribute
    extended_regex : bool, default=True
        If True, allows for "|, &, ~" characters to form long regex patterns.

    Returns
    -------
    _pat : list/pd.Index
        If K is from `pandas`, the result is a `pd.Index` object, else a list
    """
    instance_check(pat, str)

    if extended_regex:
        # split pat into different terms.
        terms = list(map(str.strip, re.split("[&|]", pat)))
        operators = re.findall("[&|]", pat)
        if len(terms) == 0:
            return Index([])
        else:
            grpres = [_foreach_flexterm(t, K) for t in terms]
            # combine
            full = grpres[0]
            for mg, op in zip(grpres[1:], operators):
                if op == "&":
                    full = intersect(full, mg)
                elif op == "|":
                    full = union(full, mg)
            return full
    else:
        return _choose_regex(pat, K)
