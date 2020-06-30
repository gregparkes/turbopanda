#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling string operationss."""

import re
import itertools as it
import numpy as np
from typing import Tuple, Union, Iterable, List
from pandas import DataFrame, Index, Series

from turbopanda.utils import set_like, belongs, instance_check

__all__ = ('patproduct', 'string_replace', 'reformat', 'shorten')


def patproduct(pat: str, *args: Iterable) -> List[str]:
    """Creates a list of strings following a certain pattern.

    Uses the old C-style of string formatting, newer style of str.format not
        currently supported.

    This is useful in the case you quickly want to produce a patterned index
    or default column names for a pandas.DataFrame index.

    Parameters
    ----------
    pat : str
        A formattable string to accept arguments
    args : list of options
        Values to insert into the pattern string

    Returns
    -------
    prod : list of str
        A list of the product of the patterns

    Examples
    --------
    A basic example would be to create default column names
    >>> import turbopanda as turb
    >>> turb.str.patproduct("%s%d", ("X", "Y"), range(100))
    >>> ["X0", ..., "X99", "Y0", ..., "Y99"]
    As you can see the product of the arguments is used, another example would be:
    >>> turb.str.patproduct("%s_%s", ("repl", "quality"), ("sum", "prod"))
    >>> ["repl_sum", "repl_prod", "quality_sum", "quality_prod"]
    """
    return [pat % item for item in it.product(*args)]


def _shorten_string(s: str, approp_len: int = 15, method: str = "middle") -> str:
    instance_check(s, str)

    if len(s) <= approp_len:
        return s
    else:
        if method == "start":
            return ".." + s[-approp_len - 2:]
        elif method == "end":
            return s[:approp_len - 2] + ".."
        elif method == "middle":
            midpoint = (approp_len - 2) // 2
            return s[:midpoint] + ".." + s[-midpoint:]
        else:
            raise ValueError("method '{}' not in {}".format(method, ('middle', 'start', 'end')))


def shorten(s, newl: int = 15, method: str = "middle"):
    """Shortens a string or array of strings to length `newl`.

    Parameters
    ----------
    s : str / list of str / np.ndarray / pd.Series / pd.Index
        The string or list of strings to shorten
    newl : int, default=15
        The number of characters to preserve (5 on each side + spaces)
    method : str, default="middle"
        Choose from {'start', 'middle', 'end'}, determines where to put dots...

    Returns
    -------
    ns : str / list of str
        A shortened string or array of strings
    """
    instance_check(s, (str, list, tuple, np.ndarray, Series, Index))
    instance_check(newl, int)
    belongs(method, ("middle", "start", "end"))

    if isinstance(s, str):
        return _shorten_string(s, newl, method)
    else:
        return [_shorten_string(_s, newl, method) for _s in s]


def string_replace(strings: Union[Series, Index],
                   operations: Tuple[str, str]) -> Series:
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
