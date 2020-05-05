#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determines the merging capabilities of turbopanda."""

# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
# imports
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame, Series, concat, merge as pmerge

from ._fileio import read
# locals
from ._metapanda import MetaPanda
from ._pipe import PipeMetaPandaType
from .utils import belongs, check_list_type, \
    get_file_expanded, instance_check, intersect, union, bounds_check

# custom types
DataSetType = Union[Series, DataFrame, MetaPanda]


def _has_majority_index_overlap(df1: DataFrame, df2: DataFrame) -> bool:
    """Checks whether the indices overlap in majority. Bool"""
    ins = intersect(df1.index, df2.index)
    return ins.shape[0] > (np.mean((df1.shape[0], df2.shape[0])) // 2)


def _intersecting_pairs(sdf1: DataFrame, sdf2: DataFrame) -> DataFrame:
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


def _maximum_likelihood_pairs(pairings: DataFrame, ret_largest: bool = True):
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


def _single_merge(sdf1: DataSetType,
                  sdf2: DataSetType,
                  how: str = 'inner') -> Union[DataFrame, MetaPanda]:
    """
    Check different use cases and merge d1 and d2 together.

    Parameters
    ----------
    sdf1 : Series, DataFrame, MetaPanda
        Dataset 1. If str, reads it in as if a file.
    sdf2 : Series, DataFrame, MetaPanda
        Dataset 2. If str, reads it in as if a file.
    how : str
        How to join on concat or merge between sdf1, sdf2.
    """
    instance_check(sdf1, (Series, DataFrame, MetaPanda))
    instance_check(sdf2, (Series, DataFrame, MetaPanda))

    # both are series.
    if isinstance(sdf1, Series) and isinstance(sdf2, Series):
        # perform pd.concat on the indexes.
        return concat((sdf1, sdf2), join=how, axis=1, sort=False, copy=True)
    elif (isinstance(sdf1, DataFrame) and isinstance(sdf2, Series)) \
            or (not (not isinstance(sdf1, Series) or not isinstance(sdf2, DataFrame))):
        # join on index. TODO: This if case may produce weird behavior.
        return concat((sdf1, sdf2), join=how, axis=1, sort=False, copy=True)
    else:
        # assign attributes based on instance types, etc.
        d1 = sdf1.df_ if isinstance(sdf1, MetaPanda) else sdf1.reset_index()
        d2 = sdf2.df_ if isinstance(sdf2, MetaPanda) else sdf2.reset_index()
        n1 = sdf1.name_ if hasattr(sdf1, "name_") else "df1"
        n2 = sdf2.name_ if hasattr(sdf2, "name_") else "df2"
        new_name = n1 + "__" + n2
        s1 = sdf1.selectors_ if hasattr(sdf1, "selectors_") else {}
        s2 = sdf2.selectors_ if hasattr(sdf2, "selectors_") else {}
        m1 = sdf1.mapper_ if hasattr(sdf1, "mapper_") else {}
        m2 = sdf2.mapper_ if hasattr(sdf2, "mapper_") else {}

        # if both of the dataframes find that their INDEX overlap...
        if _has_majority_index_overlap(d1, d2):
            # simply use concat
            df_m = concat((d1, d2), sort=False, join=how, axis=1, copy=True)
        else:
            # find the best union pair
            pair, value = _maximum_likelihood_pairs(_intersecting_pairs(d1, d2))
            # find out if we have a shared parameter
            shared_param = pair[0] if pair[0] == pair[1] else None

            # handling whether we have a shared param or not.
            merge_extra = dict(how=how, suffixes=('__%s' % n1, '__%s' % n2))
            merge_shared = dict(on=shared_param) if shared_param is not None else \
                dict(left_on=pair[0], right_on=pair[1])

            # merge pandas.DataFrames together
            df_m = pmerge(d1, d2, **merge_extra, **merge_shared)

            if shared_param is None:
                df_m.drop(pair[1], axis=1, inplace=True)
            # drop any columns with 'counter' in
            if 'counter' in df_m.columns:
                df_m.drop('counter', axis=1, inplace=True)
            elif "counter__%s" % n1 in df_m.columns:
                df_m.drop("counter__%s" % n1, axis=1, inplace=True)
            elif "counter__%s" % n2 in df_m.columns:
                df_m.drop("counter__%s" % n2, axis=1, inplace=True)
            # rename
            # df_m.rename(columns=dict(zip(df_m.columns.tolist(), d1.columns.tolist())), inplace=True)

        # create a copy metapanda, and set new attributes.
        mpf = MetaPanda(df_m, name=new_name, with_clean=False, with_warnings=False)
        # tack on extras
        mpf._select = {**s1, **s2}
        mpf._mapper = {**m1, **m2}
        # generate meta columns
        mpf.update_meta()
        # pipes are not transferred over
        return mpf


def merge(mdfs: Union[str, List[DataSetType]],
          name: Optional[str] = None,
          how: str = 'inner',
          clean_pipe: Optional[PipeMetaPandaType] = None,
          verbose: int = 0):
    """Merge together K datasets.

    Merges together a series of MetaPanda objects. This is primarily different
    to pd.merge as turb.merge will AUTOMATICALLY select for each pairing of DataFrame
    which column names have the largest crossover of labels, using maximum-likelihood.

    Parameters
    ---------
    mdfs : str/pd.Series/pd.DataFrame/MetaPanda
        An ordered set of DataFrames to merge together. Must be at least 2 elements.
            If str: reads in using glob-like, must read in multiple files then merges
            If list of str: reads in each separate file then merges
            If list of DataFrame/Series/MetaPanda, simply merges them together
    name : str, optional
        A new name to give the merged dataset.
        If None, joins together the names of every dataset in `mdfs`.
    how : str, optional
        Choose from {'inner', 'outer', 'left'}, see pandas.merge for more details.
        'inner': drops rows that aren't found in all Datasets
        'outer': keeps all rows
        'left': drops rows that aren't found in first Dataset
    clean_pipe : Pipe, optional
        A set of instructions to pass to the fully-merged DataFrame once we're done.
        See turb.Pipe() for details.
    verbose : int, optional
        If greater than 0, prints out various useful debugging messages.

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

    See Also
    --------
    pd.merge : Merge DataFrame or named Series objects with a database-style join.
    """
    # check the element type of every mdf
    instance_check(mdfs, (str, list, tuple))
    instance_check(name, (type(None), str))
    belongs(how, ['left', 'inner', 'outer'])
    bounds_check(verbose, 0, 4)

    if isinstance(mdfs, str):
        """Use glob to select multiple files to merge together."""
        mdfs = read(mdfs)
    else:
        ff_ext = get_file_expanded(mdfs)
        # filter out non-dataset type elements
        mdfs = list(filter(lambda x: not isinstance(x, str), mdfs))
        if len(ff_ext) > 0:
            # if we found some glob-like files, read in every ff_ext as a metapanda and append to mdfs
            mdfs += [read(f) for f in ff_ext]

    if len(mdfs) < 2:
        raise ValueError("mdfs must be at least length 2")
    elif len(mdfs) == 2:
        nmdf = _single_merge(mdfs[0], mdfs[1], how=how)
    else:
        nmdf = mdfs[0]
        for ds in mdfs[1:]:
            nmdf = _single_merge(nmdf, ds, how=how)

    # do some additional things if the return type is a MetaPanda object.
    if check_list_type(mdfs, MetaPanda, raised=False):
        # remove duplicated columns
        non_dup_columns = union(*[ds.columns for ds in mdfs])
        nmdf._df = nmdf.df_.loc[:, ~nmdf.columns.duplicated()]

        # add on a meta_ column indicating the source of every feature.
        col_sources = concat([Series(ds.name_,
                                     index=intersect(ds.columns, non_dup_columns))
                              for ds in mdfs],
                             axis=0, sort=False)
        col_sources.name = "datasets"
        # JOIN on the column to the dataframe - otherwise it throws a bloody error
        nmdf._meta = nmdf.meta_.join(col_sources.astype("category"))
        # join together sources into a list
        nmdf.source_ = [mdf.source_ for mdf in mdfs]
        # override name if given
        if name is not None:
            nmdf.name_ = name
        # compute clean if applicable
        if clean_pipe is not None:
            nmdf.compute(clean_pipe, inplace=True)

    # return
    return nmdf
