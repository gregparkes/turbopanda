#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:37:45 2019

@author: gparkes
"""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
from pandas import merge as pmerge
from pandas import DataFrame, Series, concat
import itertools as it

# locals
from .metapanda import MetaPanda
from .utils import check_list_type, belongs

__all__ = ("merge")


def _intersecting_pairs(sdf1, sdf2):
    """
    Given two DataFrames, find all pairings of unions.
    """
    # select subsets of columns based on object type
    sdf1_cols = sdf1.columns[sdf1.dtypes.eq(object)]
    sdf2_cols = sdf2.columns[sdf2.dtypes.eq(object)]

    # calculate pairings
    pairings = tuple(it.product(sdf1_cols, sdf2_cols))
    # empty array
    arr = np.zeros((len(pairings), 2))
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


def _single_merge(sdf1, sdf2, how):
    """Check different use cases and merge d1 and d2 together."""

    # both are series.
    if isinstance(sdf1, Series) and isinstance(sdf2, Series):
        # perform pd.concat on the indexes.
        return concat((sdf1, sdf2), join='outer', axis=1, sort=False, copy=True)
    elif (isinstance(sdf1, DataFrame) and isinstance(sdf2, Series)) \
            or (isinstance(sdf1, Series) and isinstance(sdf2, DataFrame)):
        # join on index.
        return concat((sdf1, sdf2), join='outer', axis=1, sort=False, copy=True)
    elif isinstance(sdf1, DataFrame) and isinstance(sdf2, DataFrame):
        # we call reset_index to allow for merges on the index also.
        d1 = sdf1.reset_index()
        d2 = sdf2.reset_index()
        new_name = 'df1__df2'
        s1 = {}
        s2 = {}
        m1 = {}
        m2 = {}
    elif isinstance(sdf1, MetaPanda) and isinstance(sdf2, DataFrame):
        d1 = sdf1.df_
        d2 = sdf2.reset_index()
        new_name = sdf1.name_ + "__df2"
        s1 = sdf1.selectors_
        s2 = {}
        m1 = sdf1.mapper_
        m2 = {}
    elif isinstance(sdf1, DataFrame) and isinstance(sdf2, MetaPanda):
        d1 = sdf1.reset_index()
        d2 = sdf2.df_
        new_name = "df1__" + sdf2.name_
        s1 = {}
        s2 = sdf2.selectors_
        m1 = {}
        m2 = sdf2.mapper_
    elif isinstance(sdf1, MetaPanda) and isinstance(sdf2, MetaPanda):
        d1 = sdf1.df_
        d2 = sdf2.df_
        new_name = sdf1.name_ + '__' + sdf2.name_
        s1 = sdf1.selectors_
        s2 = sdf2.selectors_
        m1 = sdf1.mapper_
        m2 = sdf2.mapper_
    else:
        raise TypeError("combination of type {}:{} not recognized".format(type(sdf1), type(sdf2)))

    # find the best union pair
    pair, value = _maximum_likelihood_pairs(_intersecting_pairs(d1, d2))
    # find out if we have a shared parameter
    shared_param = pair[0] if pair[0] == pair[1] else None

    # handling whether we have a shared param or not.
    merge_extra = dict(how=how, suffixes=('_x', '_y'))
    merge_shared = dict(on=shared_param) if shared_param is not None else dict(left_on=pair[0], right_on=pair[1])

    # merge pandas.DataFrames together
    df_m = pmerge(d1, d2, **merge_extra, **merge_shared)

    if shared_param is None:
        df_m.drop(pair[1], axis=1, inplace=True)
    # drop any columns with 'counter' in
    if 'counter' in df_m.columns:
        df_m.drop('counter', axis=1, inplace=True)
    # rename
    df_m.rename(columns=dict(zip(df_m.columns.tolist(), d1.columns.tolist())), inplace=True)
    # form a MetaPanda
    mpf = MetaPanda(df_m, name=new_name)
    # tack on extras
    mpf._select = {**s1, **s2}
    mpf._mapper = {**m1, **m2}
    # generate meta columns
    mpf._define_metamaps()
    # pipes are not transferred over
    return mpf


def merge(mdfs, how: str = 'inner', clean_pipe=None):
    """
    Merges together a series of MetaPanda objects. This is primarily different
    to pd.merge as turb.merge will AUTOMATICALLY select for each pairing of DataFrame
    which column names have the largest crossover of labels, using maximum-likelihood.

    Parameters
    ---------
    mdfs : list of pd.Series/pd.DataFrame/MetaPanda
        An ordered set of DataFrames to merge together. Must be at least 2 elements.
    how : str, optional
        Choose from {'inner', 'outer', 'left'}, see pandas.merge for more details.
        'inner': drops rows that aren't found in all Datasets
        'outer': keeps all rows
        'left': drops rows that aren't found in first Dataset
    clean_pipe : Pipe/None, optional
        A set of instructions to pass to the fully-merged DataFrame once we're done.
        See turb.Pipe() for details.

    Raises
    ------
    IndexException
        If there is no intersection between the indexes of both datasets
    ValueException
        If `how` is not one of the choice options, length of `mdfs` must be at least 2

    Returns
    -------
    nmdf : MetaPanda
        The fully merged Dataset
    """
    belongs(how, ['left', 'inner', 'outer'])

    if len(mdfs) < 2:
        raise ValueError("mdfs must be at least length 2")
    elif len(mdfs) == 2:
        nmdf = _single_merge(mdfs[0], mdfs[1], how=how)
    else:
        nmdf = mdfs[0]
        for ds in mdfs[1:]:
            nmdf = _single_merge(nmdf, ds, how=how)

    if clean_pipe is not None and isinstance(nmdf, MetaPanda):
        return nmdf.compute(clean_pipe, inplace=False)
    else:
        return nmdf
