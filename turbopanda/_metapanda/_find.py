#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to finding column names in a Metapanda, through efficient Levenshtein distancing."""


from turbopanda.str import levenshtein
from turbopanda.utils import instance_check


def find(self, X: str) -> str:
    """Finds the column name closest to the search string.

    Parameters
    ----------
    X : str
        A string name close to the column name.

    Returns
    -------
    ns : str
        The actual column name closest to the search.
    """
    instance_check(X, str)

    levenshtein(self.columns, [X])
