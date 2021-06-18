#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling string operations on pandas.DataFrames and more.
We collapse down all of the different routes into a catch-all `pattern` function which will discern the pattern
from a bunch of different options.
"""
from typing import Union

import itertools as it
import re
import pandas as pd
from pandas import Index, DataFrame, Series
from numpy import ndarray
from functools import reduce

from turbopanda.utils import set_like, union, intersect, absdifference, instance_check


def _strpattern(pat: str, K: Union[list, tuple, ndarray, Index, Series]) -> Index:
    """Determines if pattern `pat` exists in list of str `K`."""
    # compile pattern - improves performance
    _p = re.compile(pat)
    # iterate over and return
    return set_like([s for s in K if re.search(_p, s)])


def _patcolumnmatch(pat: str, df: DataFrame) -> Index:
    """Use regex to match to column names in a pandas.DataFrame."""
    c_fetch = _strpattern(pat, df.columns)
    if len(c_fetch) > 0:
        return Index(c_fetch, dtype=object, name=df.columns.name, tupleize_cols=False)
    else:
        return Index([], name=df.columns.name)


def _choose_regex(pat: str, K):
    if isinstance(K, pd.DataFrame):
        return _patcolumnmatch(pat, K)
    elif isinstance(K, pd.Series):
        return _strpattern(pat, K.dropna().values)
    else:
        return _strpattern(pat, K)


def _foreach_flexterm(term: str, K):
    if term.startswith("~"):
        s = _choose_regex(term[1:], K)
        if isinstance(K, pd.DataFrame):
            res = absdifference(K.columns, s)
        else:
            res = absdifference(K, s)
    else:
        res = _choose_regex(term, K)
    return res


def _integrate_terms(a, b):
    """where a, b are packaged (term, op)"""
    t1, op = a
    t2, op2 = b
    if op == '&':
        # return a 2-tuple
        return (intersect(t1, t2), op2)
    elif op == '|':
        # return a 2-tuple
        return (union(t1, t2), op2)
    else:
        return t1


def pattern(
    pat: str,
    K: Union[list, tuple, Series, Index, DataFrame],
    extended_regex: bool = True,
):
    """Determines if pattern `pat` exists in K.

    Parameters
    ----------
    pat : str
        Regex-compliant pattern. Also supports *flexible* regex with
            NOT operators in the case of pandas.DataFrame columns, etc.
    K : list, tuple, pd.Series, pd.Index, pd.DataFrame
        The full list to select a pattern from. Accepts pandas. inputs,
            and uses the .columns attribute
    extended_regex : bool, default=True
        If True, allows for "|, &, ~" characters to form long regex patterns.

    Returns
    -------
    pat : list/pd.Index
        If K is from `pandas`, the result is a `pd.Index` object, else a list

    Examples
    --------
    When extended regex is not on, it behaves similarly to the regex engine would
    normally do:
    >>> from turbopanda.str import pattern
    >>> greetings = ["hello", "bellow", "mellow", "swellow"]
    >>> pattern("ellow", greetings)
    >>> ["bellow", "mellow", "swellow"]
    With extended regex, you can use intersection and union operations in between
    each regex call (space optional):
    >>> pattern("^he | ^b", greetings)
    >>> ["hello", "bellow"]
    It essentially treats each block separated by | or & as a separate regex search.
    When a pandas.DataFrame is passed, the search is applied to the *column names*, and
    types can be search for:
    >>> pattern("float & pixel_", greetings)
    >>> []
    Here only float columns AND that contain 'pixel_' in the column name are selected
    """
    instance_check(pat, str)
    if len(K) == 0:
        return []

    if extended_regex:
        # split pat into different terms.
        terms = list(map(str.strip, re.split("[&|]", pat)))
        operators = re.findall("[&|]", pat)
        if len(terms) == 0:
            return Index([])
        elif len(terms) == 1:
            # handle singular case with no split terms found.
            return _foreach_flexterm(terms[0], K)
        else:
            # iterate over and reduce.
            grpres = [_foreach_flexterm(t, K) for t in terms]
            # reduce using intersect or union
            full = reduce(_integrate_terms, it.zip_longest(grpres, operators))
            return full
    else:
        return _choose_regex(pat, K)
