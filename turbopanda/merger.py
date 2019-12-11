#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:37:45 2019

@author: gparkes
"""
import numpy as np
from pandas import merge as pmerge
from pandas import DataFrame
import itertools as it

from .metapanda import MetaPanda
from .utils import check_list_type

__all__ = ["merge"]


def _intersecting_pairs(sdf1, sdf2):
    """
    Given two MetaPandas, find all pairings of unions.
    """
    # select subsets of columns based on object type
    if isinstance(sdf1, MetaPanda):
        sdf1_cols = sdf1.view(object)
        sdf1 = sdf1.df_
    else:
        sdf1_cols = sdf1.columns[sdf1.dtypes.eq(object)]
    if isinstance(sdf2, MetaPanda):
        sdf2_cols = sdf2.view(object)
        sdf2 = sdf2.df_
    else:
        sdf2_cols = sdf2.columns[sdf2.dtypes.eq(object)]

    # calculate pairings
    pairings = list(it.product(sdf1_cols, sdf2_cols))
    # empty array
    arr = np.zeros((len(pairings),2))
    for i, (p1, p2) in enumerate(pairings):
        r1 = sdf1[p1].dropna().drop_duplicates()
        r2 = sdf2[p2].dropna().drop_duplicates()
        arr[i, 0] = r1.isin(r2).sum()
        arr[i, 1] = r2.isin(r1).sum()
    return DataFrame(arr, columns=["p1 in p2", "p2 in p1"], index=pairings)


def _maximum_likelihood_pairs(pairings, ret_largest=True):
    """
    Given a pairings, choose the maximum likely pairing.
    """
    pm = pairings.mean(axis=1)
    if pm.gt(0).sum() == 0:
        raise ValueError("There is no crossover between these datasets")
    elif pm.gt(0).sum() == 1 or ret_largest:
        return pm.idxmax(), pm.max()
    else:
        return pm[pm.gt(0)]


def _single_merge(sdf1, sdf2):
    # find the best union pair
    pair, value = _maximum_likelihood_pairs(_intersecting_pairs(sdf1, sdf2))
    # find out if we have a shared parameter
    shared_param = pair[0] if pair[0] == pair[1] else None
    if shared_param is not None:
        # attempt to merge together.
        pdm = MetaPanda(pmerge(sdf1.df_, sdf2.df_, how="inner", on=shared_param), key=shared_param)
    else:
        pdm = MetaPanda(
                pmerge(sdf1.df_, sdf2.df_, how="inner", left_on=pair[0], right_on=pair[1]).drop(pair[1], axis=1),
        )
    return pdm


def merge(sdfs, clean_pipe=None):
    """
    Merges together a series of MetaPanda objects.

    Parameters
    ---------
    sdfs : list of turb.MetaPanda
        An ordered set of MetaPandas to merge together. We use 'inner' by default.
    clean_pipe : pipeline
        A set of instructions to pass to the fully-merged dataframe once we're done

    Returns
    -------
    nsdf : turb.MetaPanda
        The fully merged dataset
    """
    check_list_type(sdfs, MetaPanda)

    nsdf = sdfs[0]
    for ds in sdfs[1:]:
        nsdf = _single_merge(nsdf, ds)
    if clean_pipe is not None:
        return nsdf.compute(clean_pipe, inplace=False)
    else:
        return nsdf
