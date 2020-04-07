#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling dictionaries and sets."""

import itertools as it
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

from pandas import Index, Series

from ._error_raise import instance_check

SetLike = Union[type(None), str, Set, Index, List, Tuple, Series]

__all__ = ("dictsplit", "dict_to_tuple", "dictzip", "dictmap", "dictchunk", "join", 'pairwise',
           "set_like", "union", "intersect", "difference", 'dictcopy')

""" DICTIONARY CONVENIENCE """


def dict_to_tuple(d: Dict) -> Tuple:
    """Converts a dictionary to a 2-tuple.

    Parameters
    ----------
    d : dict
        The dictionary

    Returns
    -------
    t : tuple
        The tuplized dictionary
    """
    return tuple((a, b) for a, b in d.items())


def dictsplit(d: Dict):
    """Splits a dictionary into two tuples."""
    return tuple(d.keys()), tuple(d.values())


def dictzip(a: Iterable, b: Iterable) -> Dict:
    """Map together a, b to make {a: b}.

    .. note:: a and b do not need to be the same length. Nones are filled where empty.

    Parameters
    ----------
    a : list/tuple
        A sequence of some kind
    b : list/tuple
        A sequence of some kind

    Returns
    -------
    d : dict
        Mapped dictionary
    """
    return dict(it.zip_longest(a, b))


def dictcopy(d1: Dict, d2: Dict):
    """Updates d2 to d1 using copy rather than direct update.

    This can be thought of as an `update` with copy.

    Note that values in d2 will override any existing same-key in d1.

    Parameters
    ----------
    d1 : dict
        first dictionary
    d2 : dict
        second dictionary

    Returns
    -------
    d3 : dict
        Result dictionary
    """
    X = d1.copy()
    X.update(d2)
    return X


def dictmap(a: Iterable, b: Callable) -> Dict:
    """Map together a with the result of function b to make {a: b}.

    b must return something that can be placed in d.

    Parameters
    ----------
    a : list/tuple
        A sequence of some kind
    b : function
        A function b(y) returning something to be placed in d

    Returns
    -------
    d : dict
        Mapped dictionary
    """
    return dict(it.zip_longest(a, map(b, a)))


def dictchunk(d: Dict, k: int = 1) -> List:
    """Divides dictionary d into k-sized list chunks.

    The list may not be in the order given in the dictionary.

    Parameters
    ----------
    d : dict
        A given dictionary, must have multiple elements.
    k : int, optional
        The size of each list-chunk

    Raises
    ------
    TypeError
        When k > length of d - not valid

    Returns
    -------
    L : list of dict
        A list containing elements from d into k-chunks

    Examples
    --------
    Given dictionary:
    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> print(dictchunk(d, 1))
    [{'a': 1}, {'b': 2}, {'c': 3}]
    >>> print(dictchunk(d, 2))
    [{'a': 1, 'b': 2}, {'c': 3}]
    """
    if k > len(d):
        raise ValueError("k cannot be greater than the length of d")
    elif k == 1:
        return [{a: d[a]} for a in d]
    else:
        split_a, split_b = dictsplit(d)
        return [
            dict(it.zip_longest(
                it.islice(split_a, i, i+k), it.islice(split_b, i, i+k)
            )) for i in range(0, len(d), k)
        ]


def join(*pipes) -> List:
    """Perform it.chain.from_iterable on iterables.

    Does not accept strings, and returns them as is. None arguments are ignored, as well as list components
        with no arguments inside them.

    Parameters
    ----------
    pipes : tuple, args
        A list of arguments, jumbled together. deploys it.chain.from_iterable.

    Warnings
    --------
    UserWarning
        Raised when the number of arguments in pipes < 2

    Returns
    -------
    l : list
        Returns the joined list
    """
    # filter out None elements.
    if len(pipes) == 1 and isinstance(pipes[0], str):
        return list(pipes)

    _p = list(filter(None.__ne__, pipes))
    _p = list(filter(lambda y: len(y) > 0, _p))
    # use itertools to chain together elements.
    return list(it.chain.from_iterable(_p))


""" SET LIKE OPERATIONS """


def set_like(x: SetLike = None) -> Index:
    """
    Convert x to something unique, set-like.

    Parameters
    ----------
    x : str, list, tuple, pd.Series, set, pd.Index, optional
        A variable that can be made set-like.
        strings are wrapped.

    Returns
    -------
    y : pd.Index
        Set-like result.
    """
    acc_types = (type(None), str, list, tuple, Series, Index, set)

    instance_check(x, acc_types)

    if x is None:
        return Index([])
    if isinstance(x, str):
        return Index([x])
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return Index([])
        else:
            return Index(sorted(set(x), key=x.index))
    elif isinstance(x, (Series, Index)):
        return Index(x.dropna().unique(), name=x.name) if x.shape[0] > 0 else Index([])
    elif isinstance(x, set):
        return Index(sorted(x, key=list(x).index))
    else:
        raise TypeError("in `set_like`: `x` must be in {}, not of type {}".format(acc_types, type(x)))


def union(*args: SetLike) -> Index:
    """Performs set union all passed arguments, whatever type they are.

    Parameters
    ----------
    args : list, tuple, pd.Series, set, pd.Index
        k List-like arguments

    Raises
    ------
    ValueError
        There must be at least two arguments

    Returns
    -------
    U : pd.Index
        Union between a | b | ... | k
    """
    if len(args) == 0:
        raise ValueError('no arguments passed')
    elif len(args) == 1:
        return set_like(args[0])
    else:
        a = set_like(args[0])
        for b in args[1:]:
            a |= set_like(b)
        return a


def intersect(*args: SetLike) -> Index:
    """Performs set intersect all passed arguments, whatever type they are.

    Parameters
    ----------
    args : list, tuple, pd.Series, set, pd.Index
        k List-like arguments

    Returns
    -------
    I : pd.Index
        Intersect between a & b & ... & k
    """
    if len(args) == 0:
        raise ValueError('no arguments passed')
    elif len(args) == 1:
        return set_like(args[0])
    else:
        a = set_like(args[0])
        for b in args[1:]:
            a &= set_like(b)
        return a


def difference(a: SetLike, b: SetLike) -> Index:
    """
    Performs set symmetric difference on a and b, whatever type they are.

    Parameters
    ----------
    a : list, tuple, pd.Series, set, pd.Index
        List-like a
    b : list, tuple, pd.Series, set, pd.Index
        List-like b

    Returns
    -------
    c : pd.Index
        Symmetric difference between a & b
    """
    return set_like(a).symmetric_difference(set_like(b))


def pairwise(f: Callable, x: SetLike, *args, **kwargs) -> List:
    """Conduct a pairwise operation on a list of elements, receiving them in pairs.

    e.g for list x = [1, 2, 3] we conduct:
        for i in range(3):
            for j in range(i, 3):
                operation[i, j]

    Parameters
    ----------
    f : function
        Receives two arguments and returns something
    x : list-like
        A list of strings, parameters etc, to pass to f, as f(x1, x2)
    *args : list
        Additional arguments to pass to f(x1, x2)
    **kwargs : dict
        Additional keyword arguments to pass to f(x1, x2)

    Returns
    -------
    y : list-like
        A list of the return elements from f(x1, x2, *args, **kwargs)
    """
    y = []
    pairs = it.combinations(x, 2)
    for p1, p2 in pairs:
        y.append(f(p1, p2, *args, **kwargs))
    return join(y)
