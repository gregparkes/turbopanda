#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:55:38 2019

@author: gparkes
"""

import warnings
import functools
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
                 key=None,
                 mode="instant",
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
        key : None, str
            Defines the primary key (unique identifier), if None does nothing.
        mode : str
            Choose from ['instant', 'delay']
            If instant, executes all functions immediately inplace
            If delay, builds a task graph and then executes inplace when 'compute()' is called
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
        self.mode_ = mode
        self._select = {}
        self._pipe = []


    ############################ OVERRIDEN OPERATIONS #######################################


    def _actionable(function):
        @functools.wraps(function)
        def new_function(self, *args, **kwargs):
            if self.mode_ == "delay":
                self._pipe.append((function.__name__, args, kwargs))
            else:
                function(self, *args, **kwargs)
                return self


    def _rename_columns(self, old, new):
        self.df_.rename(columns=dict(zip(old, new)), inplace=True)
        self.meta_.rename(index=dict(zip(old, new)), inplace=True)


    def _drop_columns(self, select):
        if select.size > 0:
            self.df_.drop(select, axis=1, inplace=True)
            self.meta_.drop(select, axis=0, inplace=True)


    def _add_pipe(self, fn, *fargs, **fkwargs):
        self._pipe.append((fn, fargs, fkwargs))


    def _apply_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_, fn):
            f = getattr(self.df_, fn)
            ndf = f(*fargs, **fkwargs)
            if isinstance(ndf, pd.DataFrame):
                self.df_ = ndf
                return self
            else:
                raise TypeError("function '{}' did not return pd.DataFrame".format(fn))
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.* API".format(fn))


    def __getitem__(self, selector):
        sel = self.view(selector)
        if sel.size > 0:
            return self.df_[sel].squeeze()


    def __delitem__(self, selector):
        # drops columns inplace
        sel = self.view(selector)
        if sel.shape[0] > 0:
            self.df_.drop(sel, axis=1, inplace=True)
            # drop meta
            self.meta_.drop(sel, axis=0, inplace=True)


    def __repr__(self):
        return "MetaPanda({}(n={}, p={}, mem={}), mode='{}')".format(self.name_,
            self.df_.shape[0], self.df_.shape[1], self.memory_, self.mode_)


    ############################### PROPERTIES ##############################################


    @property
    def df_(self):
        return self._df
    @df_.setter
    def df_(self, df):
        if isinstance(df, pd.DataFrame):
            # cleans and preprocesses dataframe
            self._df = dataframe_clean(df, self._cat_thresh, self._def_remove_single_col)
            self._meta = construct_meta(self._df)
            if "colnames" not in self._df.columns:
                self._df.columns.name = "colnames"
            if "counter" not in self._df.columns:
                self._df.index.name = "counter"
        else:
            raise TypeError("'df' must be of type [pd.DataFrame]")


    @property
    def meta_(self):
        return self._meta
    @meta_.setter
    def meta_(self, meta):
        if isinstance(meta, pd.DataFrame):
            self._meta = dataframe_clean(meta, self._cat_thresh, self._def_remove_single_col)
            self._meta.index.name = "colnames"
        else:
            raise TypeError("'meta' must be of type [pd.DataFrame]")


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


    @property
    def mode_(self):
        return self._mode
    @mode_.setter
    def mode_(self, mode):
        if mode in ["instant", "delay"]:
            self._mode = mode
        else:
            raise ValueError("'mode' must be ['instant', 'delay'], not '{}'".format(mode))


    ################################ FUNCTIONS ###################################################


    def head(self, k=5):
        """
        Simply displays the top k rows of the pandas.DataFrame. This function is not affected by
        the 'mode' parameter.

        Parameters
        --------
        k : int
            Must be 0 < k < n.

        Returns
        -------
        ndf_ : pandas.DataFrame
            First k rows of self.df_
        """
        return self.df_.head(k)


    def view(self, *selector):
        """
        Select merely returns the columns of interest selected using this selector. This function
        is not affected by the 'mode' parameter.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        ------
        sel : list
            The list of column names selected, or empty
        """
        sel = get_selector(self.df_, self.meta_, self._select, selector, select_join="OR")
        if sel.shape[0] == 0:
            warnings.warn("selection: '{}' was empty, no columns selected.".format(selector), UserWarning)
        return sel


    def view_not(self, *selector):
        """
        Select merely returns the columns of interest NOT selected using this selector. This function
        is not affected by the 'mode' parameter.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        ------
        sel : list
            The list of column names NOT selected, or empty
        """
        return self.df_.columns.drop(self.view(*selector))


    @_actionable
    def apply(self, f_name, *f_args, **f_kwargs):
        """
        Applies a pandas.DataFrame.* function to the MetaPanda dataset.

        e.g mdf.apply("groupby", ["counter","refseq_id"], as_index=False)
            applies self.df_.groupby() to data and return value is stored in df_
            assumes pandas.DataFrame is returned.

        Parameters
        -------
        f_name : str
            The name of the function
        f_args : list/tuple
            Arguments to pass to the function
        f_kwargs : dict
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_function(f_name, *f_args, **f_kwargs)
        return self


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
        self._drop_columns(self.view(*selector))
        return self


    def cache(self, name, *selector):
        """
        Saves a 'selector' to use at a later date. This can be useful if you
        wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        This function is not affected by the 'mode' parameter.

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
        if name in self._select:
            warnings.warn("cache name '{}' already exists in .cache, overriding".format(name), UserWarning)
        self._select[name] = selector
        return self


    def keys(self, prim_key=None, second_key=None):
        """
        Defines and creates primary and secondary ID keys to the dataset. A primary key is a
        unique identifier to each row within a dataset as defined by third normal
        form theory.

        This function is not affected by the 'mode' parameter.

        Parameters
        --------
        prim_key : str, None
            If None, creates a primary key from 'counter', else assigns column name
            as this key.
        second_key : None, str, list/tuple
            Assigns one or more columns to be 'secondary' keys for other MetaPanda
            datasets.

        Returns
        -------
        self
        """
        return NotImplemented


    def multi_cache(self, **caches):
        """
        Saves a group of 'selectors' to use at a later date. This can be useful
        if you wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        This function is not affected by the 'mode' parameter.

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


    def rename(self, ops, selector=None):
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

        Returns
        -------
        self or pd.Index
        """
        # check ops is right format
        is_twotuple(ops)
        curr_cols = sel_cols = self.view(*selector) if selector is not None else self.df_.columns

        for op in ops:
            curr_cols = curr_cols.str.replace(*op)

        self._rename_columns(sel_cols, curr_cols)
        return self


    def add_prefix(self, pref, selector=None):
        """
        Adds a prefix to all of the columns or selected columns.

        Parameters
        -------
        pref : str
            The prefix to add
        selector : None, str, or tuple args
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
        apply : bool
            If true, performs operation inplace on dataset, else returns the
            new appearance only.

        Returns
        ------
        self
        """
        sel_cols = self.view(*selector) if selector is not None else self.df_.columns
        curr_cols = pref + sel_cols

        # set to df_ and meta_
        self._rename_columns(sel_cols, curr_cols)
        return self


    def add_suffix(self, suf, selector=None):
        """
        Adds a suffix to all of the columns or selected columns.

        Parameters
        -------
        suf : str
            The prefix to add
        selector : None, str, or tuple args
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
        apply : bool
            If true, performs operation inplace on dataset, else returns the
            new appearance only.

        Returns
        ------
        self
        """
        sel_cols = self.view(*selector) if selector is not None else self.df_.columns
        curr_cols = sel_cols + suf

        # set to df_ and meta_
        self._rename_columns(sel_cols, curr_cols)
        return self


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


    def transform(self, function, selector=None, *args):
        """
        Performs an inplace transformation to a group of columns within the df_
        attribute.

        Parameters
        -------
        function : function
            A function taking the pd.Series x_i as input and returning pd.Series y_i as output
        selector : None or str
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename
            If None, applies the function to all columns.
        args : list
            Additional arguments to pass to function(x, *args)

        Returns
        -------
        self
        """
        # perform inplace
        selection = self.view(selector) if selector is not None else self.df_.columns
        # modify
        if callable(function) and selection.shape[0] > 0:
            self.df_.loc[:, selection] = self.df_.loc[:, selection].transform(lambda x: function(x, *args))
        return self


    def multi_transform(self, ops):
        """
        Performs multiple inplace transformations to a group of columns within the df_
        attribute.

        Parameters
        -------
        ops : list of 2-tuple
            Containing:
                1. function - A function taking the pd.Series x_i as input and returning pd.Series y_i as output
                2. selector - Contains either types, meta column names, column names or regex-compliant strings
                    Allows user to specify subset to rename
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


    def sort_columns(self, by="alphabet", ascending=True):
        """
        Sort by any categorical meta_ column or "alphabet".

        Parameters
        -------
        by : str or list/tuple of str
            Sorts the columns by order of what terms are given. "alphabet" refers
            to the column name alphabetical sorting. Choose a meta_ column categorical
            to sort by.
        ascending : bool or list/tuple of bool
            Sort ascending vs descending. Specify list for multiple sort orders.

        Returns
        -------
        self
        """
        if isinstance(by, str):
            by = "colnames" if by == "alphabet" else by
            if (by in self.meta_) or (by == "colnames"):
                # sort the meta
                self.meta_.sort_values(by=by, axis=0, ascending=ascending, inplace=True)
                # sort the df
                self.df_ = self.df_.reindex(self.meta_.index, axis=1)
        elif isinstance(by, (list, tuple)):
            if "alphabet" in by:
                by[by.index("alphabet")] = "colnames"
            if isinstance(ascending, bool):
                # turn into a list with that value
                ascending=[ascending]*len(by)
            if len(by) != len(ascending):
                raise ValueError("the length of 'by' {} must equal the length of 'ascending' {}".format(len(by),len(ascending)))
            if all([(col in self.meta_) or (col == "colnames") for col in by]):
                # sort meta
                self.meta_.sort_values(by=by, axis=0, ascending=ascending, inplace=True)
                # sort df
                self._df = self.df_.reindex(self.meta_.index, axis=1)
        else:
            raise TypeError("'by' must be of type [str, list, tuple], not {}".format(type(by)))


    def expand(self, column, sep=","):
        """
        Expands out a 'stacked' id column to a longer-form dataframe, and re-merging
        the data back in.

        Parameters
        -------
        column : str
            The name of the column to expand, must be of datatype [object]
        sep : str
            The separating string to use.

        Returns
        -------
        self
        """
        if column not in self.df_.columns:
            raise ValueError("column '{}' not found in df".format(column))
        if not self.meta_.loc[column, "potential_stacker"]:
            raise ValueError("column '{}' not found to be stackable".format(column))

        self._df = pd.merge(
            # expand out id column
            self.df_[column].str.strip().str.split(sep).explode(),
            self.df_.dropna(subset=[column]).drop(column, axis=1),
            left_index=True, right_index=True
        )
        self._df.columns.name = "colnames"
        return self


    def shrink(self, column, sep=","):
        """
        Shrinks  down a 'duplicated' id column to a shorter-form dataframe, and re-merging
        the data back in.

        Parameters
        -------
        column : str
            The name of the duplicated column to shrink, must be of datatype [object]
        sep : str
            The separating string to add.

        Returns
        -------
        self
        """
        if column not in self.df_.columns:
            raise ValueError("column '{}' not found in df".format(column))
        if self.meta_.loc[column, "is_unique"]:
            raise ValueError("column '{}' found to be unique".format(column))

        # no changes made to columns, use hidden df
        self._df = pd.merge(
            # shrink down id column
            self.df_.groupby("counter")[column].apply(lambda x: x.str.cat(sep=sep)),
            self.df_.drop(column, axis=1).drop_duplicates(keep="first").set_index("counter"),
            left_index=True, right_index=True
        )
        self._df.columns.name = "colnames"
        return self


    def split_categories(self, column, sep=",", renames=None):
        """
        Splits a column into N categorical variables to be associated with df_.

        Parameters
        -------
        column : str
            The name of the column to split, must be of datatype [object], and contain values sep inside
        sep : str
            The separating string to add.
        renames : None or list of str
            If list of str, must be the same dimension as expanded columns
        """
        if column not in self.df_.columns:
            raise ValueError("column '{}' not found in df".format(column))

        exp = self.df_[column].str.strip().str.split(sep, expand=True)
        # calculate column names
        if renames is None:
            cnames = ["cat%d" % (i+1) for i in range(exp.shape[1])]
        else:
            cnames = renames if len(renames) == exp.shape[1] else ["cat%d" % (i+1) for i in range(exp.shape[1])]

        self._df = self.df_.join(
            exp.rename(columns=dict(zip(range(exp.shape[1]), cnames)))
        )
        self._df.columns.name = "colnames"
        return self


    def melt(self, id_vars=["counter"], *args, **kwargs):
        """
        Performs a 'melt' operation on the DataFrame, with some minor adjustments to
        maintain the integrity of the MetaPandaframe.

        Parameters
        --------
        id_vars : list, optional
            Column(s) to use as identifier variables.
        args, kwargs : list/dict
            arguments to pass to pandas.DataFrame.melt

        Returns
        -------
        self
        """
        if isinstance(id_vars, list) or id_vars is None:
            if "counter" not in id_vars:
                id_vars.insert(0, "counter")
            # modify df by resetting and melting
            self.df_ = (self.df_.reset_index()
                .melt(id_vars=id_vars, *args, **kwargs))
        return self


    def compute(self):
        """
        Computes tasks in-order depending on the pipeline cached.

        Does nothing if mode == 'instant'

        Returns
        -------
        self
        """
        return NotImplemented


    def write(self, filename=None, with_meta=True, *args, **kwargs):
        """
        Saves a MetaPanda to disk.

        This function is not affected by the 'mode' parameter.

        Parameters
        -------
        filename : str
            The name of the file to save, or None it will use the name found in MetaPanda
        with_meta : bool
            If true, saves metafile also, else doesn't
        *args : list
            Arguments to pass to pd.to_csv
        **kwargs : dict
            Keywords to pass to pd.to_csv
        """
        if filename is not None:
            self.df_.to_csv(filename, sep=",",*args,**kwargs)
            if with_meta:
                # uses the name stored in mp
                dsplit = filename.rsplit("/",1)
                if len(dsplit) == 1:
                    self.meta_.to_csv(dsplit[0].split(".")[0]+"__meta.csv",sep=",")
                else:
                    directory, name = dsplit
                    self.meta_.to_csv(directory+"/"+name.split(".")[0]+"__meta.csv",sep=",")
        else:
            self.df_.to_csv(self.name_+".csv", sep=",",*args,**kwargs)
            if with_meta:
                self.meta_.to_csv(self.name_+"__meta.csv",sep=",")


    _actionable = staticmethod(_actionable)
