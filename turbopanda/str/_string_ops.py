#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling string operationss."""

import re
from typing import Tuple, Union

from pandas import DataFrame, Index, Series

from turbopanda.utils import set_like

__all__ = ('strpattern', 'string_replace', 'reformat')


def strpattern(pat, K):
    """Determines if pattern `pat` exists in list of str `K`."""
    return set_like([s for s in K if re.search(pat, s)])


def string_replace(strings: Union[Series, Index], operations: Tuple[str, str]) -> Series:
    """ Performs all replace operations on the string inplace """
    for op in operations:
        strings = strings.str.replace(*op)
    return strings


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
