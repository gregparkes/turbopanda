#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:55:38 2019

@author: gparkes
"""

import warnings
import pandas as pd

from .utils import *
from .selection import get_selector, categorize
from .analyze import agglomerate
from .metadata import construct_meta


class MetaPanda(object):
    """
    A pandas.DataFrame, but with a few extra goodies. Contains meta-functionality
    for the pandas.DataFrame, including complex handling of grouped-columns.
    """

    def __init__(self,
                 dataset,
                 name="DataSet",
                 cat_thresh=20,
                 default_remove_single_col=True):
        """
        Creates a Meta DataFrame with the raw data and parameterization of
        the dataframe by its grouped columns.

        Parameters
        -------
        dataset : pd.DataFrame
            The raw dataset to create as a metadataframe.
        name : str
            Gives the metaframe a name, which comes into play with merging, for instance
        cat_thresh : int
            The threshold until which 'category' variables are not created
        default_remove_one_column : bool
            Decides whether to drop columns with a single unique value in (default True)
        """

        self._cat_thresh = cat_thresh
        self._def_remove_single_col = default_remove_single_col
        # set using property
        self.df_ = dataset
        self.name_ = name
        self._select = {}


    ############################### HIDDEN FUNCTIONS ##################################


    def _remove_MultiIndex(self):
        if isinstance(self.df_.columns, pd.MultiIndex):
            indices = [n if n is not None else ("Index%d" % i) for i, n in enumerate(self.df_.columns.names)]
            self.df_.columns = pd.Index(["__".join(col) for col in self.df_.columns], name="__".join(indices))


    def _remove_spaces_in_obj_columns(self):
        for c in self.df_.columns[self.df_.dtypes.eq(object)]:
            self.df_[c] = self.df_[c].str.strip()
        # if we have an object index, strip this string also
        if self.df_.index.dtype == object:
            self.df_.index = self.df_.index.str.strip()


    def _single_merge(self, other, left_on, right_on, join_type, **kwargs):
        # determine index use
        left_ind = True if left_on is None else False
        right_ind = True if right_on is None else False
        if left_ind and not right_ind:
            ndf = pd.merge(self.df_, other.df_, left_index=True, right_on=right_on, how=join_type, **kwargs)
        elif not left_ind and right_ind:
            ndf = pd.merge(self.df_, other.df_, left_on=left_on, right_index=True, how=join_type, **kwargs)
        elif left_ind and right_ind:
            ndf = pd.merge(self.df_, other.df_, left_index=True, right_index=True, how=join_type,**kwargs)
        else:
            ndf = pd.merge(self.df_, other.df_, left_on=left_on, right_on=right_on, how=join_type,**kwargs)

        return ndf

    ############################ OVERRIDEN OPERATIONS #######################################


    def __getitem__(self, selector):
        # firstly check if 'key' is a datatype to keep!#
        sel = get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")
        return self.df_[sel].squeeze()


    def __delitem__(self, selector):
        # drops columns inplace
        sel = get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")
        if sel.shape[0] > 0:
            self.df_.drop(sel, axis=1, inplace=True)
            # drop meta
            self.meta_.drop(sel, axis=0, inplace=True)


    def __repr__(self):
        return "MetaPanda({}(n={}, p={}, mem={}))".format(self.name_,
            self.df_.shape[0], self.df_.shape[1], self.memory_)


    ############################### PROPERTIES ##############################################


    @property
    def df_(self):
        return self._df
    @df_.setter
    def df_(self, df):
        if isinstance(df, pd.DataFrame):
            # convert columns to numeric if possible
            self._df = df.apply(pd.to_numeric, errors="ignore")
            # calculate level depth
            self._levels = self._df.columns.nlevels if isinstance(self._df.columns, pd.MultiIndex) else 1
            # if multi-column, concatenate to a single column.
            self._remove_MultiIndex()
            # perform categorization
            categorize(self._df, self._cat_thresh, self._def_remove_single_col)
            # strip column names of spaces either side
            self._df.columns = self._df.columns.str.strip()
            # strip spaces within text-based features
            self._remove_spaces_in_obj_columns()
            # remove spaces/tabs within the column name.
            self._df.columns = self._df.columns.str.replace(" ", "_").str.replace("\t","_").str.replace("-","")
            # sort index
            self._df.sort_index(inplace=True)
            # construct metadata on columns
            self._meta = construct_meta(self.df_)
        else:
            raise TypeError("'df' must be of type [pd.DataFrame]")


    @property
    def meta_(self):
        return self._meta


    @property
    def memory_(self):
        return "{:0.3f}MB".format(self.df_.memory_usage().sum() / 1000000)


    ################################ FUNCTIONS ###################################################


    def view(self, *selector):
        """
        Select merely returns the columns of interest selected using this selector.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        ------
        sel : list
            The list of column names selected, or empty
        """
        return get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")


    def view_not(self, *selector):
        """
        Select merely returns the columns of interest NOT selected using this selector.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        ------
        sel : list
            The list of column names NOT selected, or empty
        """
        selected_list = get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")
        return self.df_.columns.symmetric_difference(selected_list)


    def drop(self, *selector):
        """
        Given a selector or group of selectors, drop all of the columns selected within
        this group, applied to df_.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        -------
        self
        """
        # perform inplace
        selection = get_selector(self.df_, self.meta_, self._select, selector,
                                 raise_error=False, select_join="OR")
        if selection.shape[0] > 0:
            self.df_.drop(selection, axis=1, inplace=True)
            self.meta_.drop(selection, axis=0, inplace=True)
        return self


    def cache(self, name, *selector):
        """
        Saves a 'selector' to use at a later date. This can be useful if you
        wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        Parameters
        -------
        name : str
            A name to reference the selector with.
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        -------
        self
        """
        instance_check(name, str)
        if name not in self._select:
            self._select[name] = selector
        else:
            warnings.warn("cache name '{}' already exists in .cache, overriding".format(name), UserWarning)
            self._select[name] = selector
        return self


    def multi_cache(self, **caches):
        """
        Saves a group of 'selectors' to use at a later date. This can be useful
        if you wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        Parameters
        --------
        caches : dict (k, w)
            keyword: unique reference of the selector
            value: selector: str, tuple args
                 Contains either types, meta column names, column names or regex-compliant

        Returns
        -------
        self
        """
        for name, selector in caches.items():
            self.cache(name, selector)
        return self


    def rename(self, ops, selector=None, apply=True):
        """
        Renames the column names within the pandas.DataFrame in a flexible fashion.

        Parameters
        -------
        ops : list of tuple (2,)
            Where the first value of each tuple is the string to find, with its replacement
            At this stage we only accept *direct* replacements. No regex.
            Operations are performed 'in order'.
        selector : None, str, or tuple args
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
        apply : bool
            If true, performs operation immediately on dataset, else returns the
            new appearance only.

        Returns
        -------
        self
        """
        # check ops is right format
        is_twotuple(ops)
        if selector is not None:
            curr_cols = sel_cols = get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")
        else:
            curr_cols = sel_cols = self.df_.columns

        for op in ops:
            curr_cols = curr_cols.str.replace(*op)

        if apply:
            # set to df_ and meta_
            self.df_.rename(columns=dict(zip(sel_cols, curr_cols)), inplace=True)
            self.meta_.rename(index=dict(zip(sel_cols, curr_cols)), inplace=True)
            return self
        else:
            return curr_cols


    def analyze(self, functions = ["agglomerate"]):
        """
        Performs a series of analyses on the column names and how they might
        associate with each other.

        Parameters
        -------
        functions : list
            Choose from:
                'agglomerate' - uses Levenshtein edit distance to determine how
                similar features are and uses FeatureAgglomeration to label each
                subgroup.

        Returns
        ------
        self
        """
        if "agglomerate" in functions:
            self.meta_["agglomerate"] = agglomerate(self.df_.columns)
        return self


    def merge(self, others, ons, how="inner", **kwargs):
        """
        Chains together database merging operations applied on to this MetaPanda object
        this object is the left-most.

        Performs changes to self.df_ and self.meta_.

        Parameters
        -------
        others : MetaPanda or list/tuple of MetaPanda
            A single or multiple MetaPanda object to integrate
        ons : str/list/tuple
            Column name(s) to join with, INCLUDING this MetaPanda's column name,
            if None, uses the index. List size must be p+1 where p is the number of
            other MetaPanda frames.
        how : str/list/tuple
            For each merge, determines which join to use. By default does inner
            join on all. Choose from [inner, outer, left, right]
        kwargs : dict
            Additional keywords to pass to pd.merge

        Returns
        -------
        self
        """
        # perform checks
        instance_check(ons, (list, tuple))
        # check to make sure ons is length of others + 1
        if isinstance(others, MetaPanda):
            ndf = self._single_merge(others, left_on=ons[0], right_on=ons[1],
                                     join_type=how, **kwargs)
            # append string names
            ndf.columns = attach_name(self, others)
            return ndf
        elif isinstance(others, (list, tuple)):
            check_list_type(others, MetaPanda)
            raise NotImplementedError

