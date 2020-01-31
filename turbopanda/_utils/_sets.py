#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling dictionaries and sets."""

import itertools as it
from typing import Dict, Tuple, Iterable, Callable, Any, List, Union, Set, Optional
from pandas import Index, Series


SetLike = Union[type(None), str, Set, Index, List, Tuple, Series]


__all__ = ("dict_to_tuple", "dictzip", "dictmap", "join", 'pairwise',
           "set_like", "union", "intersect", "difference", 'interacting_set')


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


def join(*pipes: Optional[Iterable[Any]]) -> List:
    """Perform it.chain.from_iterable on iterables."""
    # filter out None elements.
    pipes = list(filter(None.__ne__, pipes))
    # use itertools to chain together elements.
    return list(it.chain.from_iterable(pipes))


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
    if x is None:
        return Index([])
    if isinstance(x, str):
        return Index([x])
    if isinstance(x, (list, tuple)):
        return Index(set(x))
    elif isinstance(x, (Series, Index)):
        return Index(x.dropna().unique(), name=x.name)
    elif isinstance(x, set):
        return Index(x)
    else:
        raise TypeError(
            "x must be in {}, not of type {}".format(
                ['None', 'str', 'list', 'tuple', 'pd.Series', 'pd.Index', 'set'], type(x)))


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


def interacting_set(sets):
    """
    Given a list of pd.Index, calculates whether any of the values are shared
    between any of the indexes.
    """
    union_l = []
    # generate a list of potential interactions.
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            interact = intersect(sets[i], sets[j])
            union_l.append(interact)

    return union(*union_l)


def pairwise(f: Callable, x: List[Any]):
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
        A list of strings, parameters etc, to pass to f

    Returns
    -------
    y : list-like
        A list of the return elements from f(x)
    """
    y = []
    pairs = it.combinations(x, 2)
    for p1, p2 in pairs:
        y.append(f(p1, p2))
    return join(y)
