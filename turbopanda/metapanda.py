#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:55:38 2019

@author: gparkes
"""

import warnings
import pandas as pd

from .utils import *
from .selection import get_selector
from .analyze import agglomerate, intersection_grid, dataframe_clean, dist
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
            # cleans and preprocesses dataframe
            self._df = dataframe_clean(df, self._cat_thresh, self._def_remove_single_col)
            # construct metadata on columns
            self._meta = construct_meta(self.df_)
        else:
            raise TypeError("'df' must be of type [pd.DataFrame]")


    @property
    def meta_(self):
        return self._meta


    @property
    def memory_(self):
        return "{:0.3f}MB".format(calc_mem(self.df_) + calc_mem(self.meta_))


    @property
    def name_(self):
        return self._name
    @name_.setter
    def name_(self, n):
        if isinstance(n, str):
            self._name = n
        else:
            raise TypeError("'name_' must be of type str")


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
            if isinstance(selector, (tuple, list)) and len(selector) > 1:
                self.cache(name, *selector)
            else:
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
            Choose any combination of:
                'agglomerate' - uses Levenshtein edit distance to determine how
                    similar features are and uses FeatureAgglomeration to label each
                    subgroup.
                'approx_dist' - estimates which statistical distribution each feature
                    belongs to.

        Returns
        ------
        self
        """
        options = ["agglomerate", "approx_dist"]

        if "agglomerate" in functions:
            self.meta_["agglomerate"] = agglomerate(self.df_.columns)
        if "approx_dist" in functions:
            self.meta_["approx_dist"] = dist(self.df_)
        return self


    def transform(self, selector, function, *args):
        """
        Performs an inplace transformation to a group of columns within the df_
        attribute.

        Parameters
        -------
        selector : str
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
        function : function
            A function taking the pd.Series x_i as input and returning pd.Series y_i as output
        args : list
            Additional arguments to pass to function(x, *args)

        Returns
        -------
        self
        """
        # perform inplace
        selection = get_selector(self.df_, self.meta_, self._select, selector,
                                 raise_error=False, select_join="OR")
        # modify
        if callable(function) and selection.shape[0] > 0:
            self.df_.loc[:, selection] = self.df_.loc[:, selection].apply(lambda x: function(x, *args))
        return self


    def multi_transform(self, ops):
        """
        Performs multiple inplace transformations to a group of columns within the df_
        attribute.

        Parameters
        -------
        ops : list of 2-tuple
            Containing:
                1. selector - Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
                2. function - A function taking the pd.Series x_i as input and returning pd.Series y_i as output

        Returns
        -------
        self
        """
        is_twotuple(ops)
        for op in ops:
            self.transform(op[0], op[1])
        return self


    def meta_map(self, name, selectors):
        """
        Maps a group of selectors into a column in the meta-information
        describing some groupings/associations between features.

        For example, your data may come from multiple sources and you wish to
        identify this within your data.

        Parameters
        --------
        name : str
            The name of this overall grouping
        selectors : list, tuple or dict
            At least 2 or more selectors identifying subgroups. If you use
            cached names, the ID of the cache name is used as an identifier, and
            likewise with a dict, else list/tuple uses default group1...groupk.

        Returns
        -------
        self
        """
        # for each selector, get the group view.
        if isinstance(selectors, dict):
            cnames = [self.view(sel) for n, sel in selectors.items()]
            names = selectors.keys()
        elif isinstance(selectors, (list, tuple)):
            cnames = [self.view(sel) for sel in selectors]
            names = [sel if sel in self._select else "group%d"%(i+1)
                        for i, sel in enumerate(selectors)]
        else:
            raise TypeError("'selectors' must be of type [list, tuple, dict]")

        igrid = intersection_grid(cnames)
        if igrid.shape[0] == 0:
            NG = pd.concat([
                    pd.Series(n, index=val) for n, val in zip(names,cnames)
                ], sort=False, axis=0)
            NG.name = name
        else:
            raise ValueError("shared terms: {} discovered for meta_map.".format(igrid))
        # merge into meta
        self.meta_[name] = NG
        # categorize
        convert_category(self.meta_, name, NG.unique())
        return self


    def write(self, filename=None, *args, **kwargs):
        """
        Saves a MetaPanda to disk.

        Parameters
        -------
        filename : str
            The name of the file to save, or None it will use the name found in MetaPanda
        *args : list
            Arguments to pass to pd.to_csv
        **kwargs : dict
            Keywords to pass to pd.to_csv
        """
        if filename is not None:
            self.df_.to_csv(filename, sep=",",*args,**kwargs)
            # uses the name stored in mp
            dsplit = filename.rsplit("/",1)
            if len(dsplit) == 1:
                self.meta_.to_csv(dsplit[0].split(".")[0]+"__meta.csv",sep=",")
            else:
                directory, name = dsplit
                self.meta_.to_csv(directory+"/"+name.split(".")[0]+"__meta.csv",sep=",")
        else:
            self.df_.to_csv(self.name_+".csv", sep=",",*args,**kwargs)
            self.meta_.to_csv(self.name_+"__meta.csv",sep=",")
