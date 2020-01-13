#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling dictionaries and sets."""

import itertools as it
from typing import Dict, Tuple, Iterable, Callable, Any, List


__all__ = ("dict_to_tuple", "dictzip", "dictmap", "join",
           "set_like", "union", "intersect", "difference")


""" DICTIONARY CONVENIENCE """


def dict_to_tuple(d: Dict) -> Tuple:
    """Converts a dictionary to a 2-tuple."""
    return tuple((a, b) for a, b in d.items())


def dictzip(a: Iterable, b: Iterable) -> Dict:
    """Map together a, b to make {a: b}.

    a and b must be the same length.

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


def join(*pipes: Iterable[Any]) -> List:
    """Perform it.chain.from_iterable on iterables."""
    return list(it.chain.from_iterable(pipes))


""" SET LIKE OPERATIONS """


def set_like(x: SetLike) -> pd.Index:
    """
    Convert x to something unique, set-like.

    Parameters
    ----------
    x : list, tuple, pd.Series, set, pd.Index
        A list of variables

    Returns
    -------
    y : pd.Index
        Set-like result.
    """
    if isinstance(x, (list, tuple)):
        return pd.Index(set(x))
    elif isinstance(x, (pd.Series, pd.Index)):
        return pd.Index(x.dropna().unique())
    elif isinstance(x, set):
        return pd.Index(x)
    else:
        raise TypeError(
            "x must be in {}, not of type {}".format(['list', 'tuple', 'pd.Series', 'pd.Index', 'set'], type(x)))


def union(*args: SetLike) -> pd.Index:
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


def intersect(*args: SetLike) -> pd.Index:
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


def difference(a: SetLike, b: SetLike) -> pd.Index:
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
