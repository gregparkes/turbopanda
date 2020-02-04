#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling utility string operationss."""

from typing import Union, Tuple, List
from pandas import Series, Index, DataFrame
from ._sets import pairwise

__all__ = ('string_replace', 'common_substring_match', 'pairwise_common_substring_matches', 'reformat')


def string_replace(strings: Union[Series, Index], operations: Tuple[str, str]) -> Series:
    """ Performs all replace operations on the string inplace """
    for op in operations:
        strings = strings.str.replace(*op)
    return strings


def common_substring_match(a: str, b: str) -> str:
    """Given two strings, find the longest common substring.

     Also known as the Longest Common Substring problem."""
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    # return the longest substring
    if match.size != 0:
        return a[match.a:match.a + match.size]
    else:
        return ""


def pairwise_common_substring_matches(array: List[str]) -> Series:
    """
    Given k strings, find the most frequent longest common substring.

    Parameters
    ----------
    array : list, tuple, pd.Index
        A list of strings

    Returns
    -------
    ser : Series
        The value counts of every pairwise common substring match
    """
    pairs = pairwise(common_substring_match, array)
    return Series(pairs).value_counts()


def reformat(s: str, df: DataFrame) -> Series:
    """Using DataFrame `df`, reformat a string column using pattern `s`.

    e.g reformat("{data_group}_{data_source}", df)
        Creates a pd.Series looking like pattern s using column data_group, data_sources input.
        Does not allow spaces within the column name.

    .. note:: currently does not allow specification for type args:
        e.g reformat("{data_number:0.3f}", df)

    Parameters
    ----------
    s : str
        The regex pattern to conform to.
    df : pd.DataFrame
        A dataset containing columns selected for in `s`.

    Returns
    -------
    ser : pd.Series
        Reformatted column.
    """
    import re
    columns = re.findall('.*?{([a-zA-Z0-9_-]+)}.*?', s)
    d = []
    for i, r in df.iterrows():
        mmap = dict(zip(columns, r[columns]))
        d.append(s.format(**mmap))
    return Series(d, index=df.index)
