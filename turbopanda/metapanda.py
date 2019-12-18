#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:55:38 2019

@author: gparkes
"""

import os
import sys
import numpy as np
import warnings
import functools
import pandas as pd
from copy import deepcopy
import json

from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy

from .utils import *
from .metadata import *
from .selection import get_selector, _type_encoder_map
from .analyze import agglomerate, intersection_grid, dist
from .pipes import clean_pipe

# hidden .py attributes

__nondelay_functions__ = [
    "head", "cache", "multi_cache", "view", "view_not",
    "analyze", "compute", "multi_compute", "write", "write_json"
]

__delay_functions__ = [
    "drop", "keep", "apply", "apply_index", "apply_columns",
    "transform", "multi_transform", "rename", "add_prefix",
    "add_suffix", "meta_map", "expand", "shrink", "sort_columns",
    "split_categories", "melt", "filter_rows"
]

__functions__ = __nondelay_functions__ + __delay_functions__


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
            The raw dataset to create as a MetaDataFrame.
        name : str
            Gives the MetaDataFrame a name, which comes into play with merging, for instance
        key : None, str
            Defines the primary key (unique identifier), if None does nothing.
        mode : str
            Choose from ['instant', 'delay']
            If instant, executes all functions immediately inplace
            If delay, builds a task graph in 'pipe_'
                and then executes inplace when 'compute()' is called
        cat_thresh : int
            The threshold until which 'category' variables are not created
        default_remove_single_col : bool
            Decides whether to drop columns with a single unique value in (default True)
        """

        self._cat_thresh = cat_thresh
        self._def_remove_single_col = default_remove_single_col
        self._with_warnings = False
        self.mode_ = mode
        self._select = {}
        self._pipe = []
        # set using property
        self.df_ = dataset
        self._key = key
        self.name_ = name

    """ ############################ STATIC FUNCTIONS ######################################## """

    def _actionable(function):
        @functools.wraps(function)
        def new_function(self, *args, **kwargs):
            if self.mode_ == "delay":
                self._pipe.append((function.__name__, args, kwargs))
            else:
                return function(self, *args, **kwargs)

        return new_function

    @classmethod
    def from_csv(cls, filename, name=None, metafile=None, key=None, *args, **kwargs):
        """
        Reads in a datafile from CSV and creates a MetaPanda object from it.

        Parameters
        -------
        filename : str
            A relative/absolute link to the file, with extension provided.
            Accepted extensions: [csv, xls, xlsx, html, json, hdf, sql]
        name : str, optional
            A custom name to use for the MetaPanda, else the filename is used
        metafile : str, optional
            An associated meta file to join into the MetaPanda, else if None,
            attempts to find the file, otherwise just creates the raw default.
        key : None, str, optional
            Sets one of the columns as the 'primary key' in the Dataset
        args : list
            Additional args to pass to pd.read_csv
        kwargs : dict
            Additional args to pass to pd.read_csv

        Returns
        ------
        mdf : MetaPanda
            A MetaPanda object.
        """
        instance_check(filename, str)
        if not os.path.isfile(filename):
            raise IOError("file at '{}' does not exist".format(filename))

        file_ext_map = {
            "csv": pd.read_csv, "xls": pd.read_excel, "xlsx": pd.read_excel,
            "html": pd.read_html, "json": pd.read_json, "hdf": pd.read_hdf,
            "sql": pd.read_sql, "XLSX": pd.read_excel
        }

        fs = filename.rsplit("/", 1)
        if len(fs) == 0:
            raise ValueError("filename '{}' not recognized".format(filename))
        elif len(fs) == 1:
            directory = "."
            fname = fs[0]
        else:
            directory, fname = fs
        # just the name without the extension
        jname, ext = fname.split(".", 1)

        df = file_ext_map[ext](filename, *args, **kwargs)
        # map to MetaPanda
        if name is not None:
            mp = cls(df, name=name, key=key)
        else:
            mp = cls(df, name=jname, key=key)
            name = "_"

        if metafile is not None:
            met = pd.read_csv(metafile, index_col=0, header=0, sep=",")
            mp.meta_ = met
        else:
            # try to find a metafile in the same directory.
            dir_files = os.listdir(directory)
            # potential combination of acceptable names to find
            combs = [jname + "__meta.csv", name + "__meta.csv"]

            for potential_name in combs:
                if potential_name in dir_files:
                    met = pd.read_csv(directory + "/" + potential_name, index_col=0, header=0, sep=",")
                    # add to mp
                    mp.meta_ = met
                    return mp

        return mp

        pass

    @classmethod
    def from_json(cls, filename):
        """
        Reads in a datafile from JSON and creates a MetaPanda object from it.
        Pipe attributes are not saved currently due to the problem of storing
        potential lambda functions.

        Parameters
        -------
        filename : str
            A relative/absolute link to the JSON file, with extension optional.

        Returns
        ------
        mdf : MetaPanda
            A MetaPanda object.
        """
        instance_check(filename, str)
        # look for attributes 'data', 'meta', 'name', 'pipe' and 'cache'
        if not os.path.isfile(filename):
            raise IOError("file at '{}' does not exist".format(filename))
        # check if ends with .json
        if not filename.endswith(".json"):
            filename += ".json"
        # read in JSON
        with open(filename, "r") as f:
            recvr = json.load(f)
        # go over attributes and assign where available
        if "data" in recvr:
            ndf = pd.DataFrame.from_dict(recvr["data"])
            ndf.index.name = "counter"
            ndf.columns.name = "colnames"
            # assign to self
            mp = cls(ndf)
        else:
            raise ValueError("column 'data' not found in MetaPandaJSON")
        if "meta" in recvr:
            met = pd.DataFrame.from_dict(recvr["meta"])
            # set to MP
            mp.meta_ = met
            # include metadata
            add_metadata(mp._df, mp._meta)
        if "name" in recvr:
            mp.name_ = recvr["name"]
        if "cache" in recvr:
            mp._select = recvr["cache"]
        return mp

    """ ############################ HIDDEN OPERATIONS ####################################### """

    def _rename_axis(self, old, new, axis=1):
        if axis == 1:
            self.df_.rename(columns=dict(zip(old, new)), inplace=True)
            self.meta_.rename(index=dict(zip(old, new)), inplace=True)
        elif axis == 0:
            self.df_.rename(index=dict(zip(old, new)), inplace=True)
        else:
            raise ValueError("axis '{}' not recognized".format(axis))

    def _drop_columns(self, select):
        if select.size > 0:
            self.df_.drop(select, axis=1, inplace=True)
            self.meta_.drop(select, axis=0, inplace=True)

    def _apply_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_, fn):
            f = getattr(self.df_, fn)
            self._df = f(*fargs, **fkwargs)
            return self
        # if we start with groupby, then we are coupling groupby with an aggregation.
        elif fn.startswith("groupby__"):
            _, fn2 = fn.split("__", 1)
            groupby_f = getattr(self.df_, "groupby")
            _grouped = groupby_f(*fargs, **fkwargs)
            if hasattr(_grouped, fn2):
                self._df = _grouped.agg(fn2)
                return self
            else:
                raise ValueError("function '{}' not recognized in pandas.DataFrame.* API".format(fn2))
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.* API".format(fn))

    def _apply_index_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_.index, fn):
            f = getattr(self.df_.index, fn)
            self.df_.index = f(*fargs, **fkwargs)
            return self
        elif hasattr(self.df_.index.str, fn):
            f = getattr(self.df_.index.str, fn)
            self.df_.index = f(*fargs, **fkwargs)
            return self
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.index.[str.]* API".format(fn))

    def _apply_column_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_.columns, fn):
            f = getattr(self.df_.columns, fn)
            self.df_.columns = f(*fargs, **fkwargs)
            self.meta_.index = f(*fargs, **fkwargs)
            return self
        elif hasattr(self.df_.columns.str, fn):
            f = getattr(self.df_.columns.str, fn)
            self.df_.columns = f(*fargs, **fkwargs)
            self.meta_.index = f(*fargs, **fkwargs)
            return self
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.columns.[str.]* API".format(fn))

    def _apply_pipe(self, pipe):
        # checks
        if len(pipe) == 0:
            warnings.warn("pipe_ empty, nothing to compute.", UserWarning)
            return
        # basic check of pipe
        if is_metapanda_pipe(pipe):
            for fn, args, kwargs in pipe:
                # check that MetaPanda has the function attribute
                if hasattr(self, fn):
                    # execute function with args and kwargs
                    getattr(self, fn)(*args, **kwargs)

    def _write_csv(self, filename, with_meta=False, *args, **kwargs):
        self.df_.to_csv(filename, sep=",", *args, **kwargs)
        if with_meta:
            # uses the name stored in mp
            dsplit = filename.rsplit("/", 1)
            if len(dsplit) == 1:
                self.meta_.to_csv(dsplit[0].split(".")[0] + "__meta.csv", sep=",")
            else:
                directory, name = dsplit
                self.meta_.to_csv(directory + "/" + name.split(".")[0] + "__meta.csv", sep=",")

    def _write_json(self, filename):
        saving_dict = {"data": self.df_.to_dict(),
                       "meta": self.meta_.drop(meta_columns_default(), axis=1).to_dict(),
                       "cache": self._select,
                       "name": self.name_
                       }
        if filename is None:
            with open(self.name_ + ".json", "w") as f:
                json.dump(saving_dict, f, separators=(",", ":"))
        else:
            with open(filename, "w") as f:
                json.dump(saving_dict, f, separators=(",", ":"))

    """ ############################## OVERIDDEN OPERATIONS ###################################### """

    def __getitem__(self, selector):
        # we take the columns that are NOT this selection, then drop to keep order.
        sel = self.view(selector)
        if sel.size > 0:
            # drop anti-selection to maintain order/sorting
            return self.df_[sel].squeeze()

    def __delitem__(self, selector):
        # drops columns inplace
        self._drop_columns(self.view(selector))

    def __enter__(self):
        self.mode_ = "delay"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mode_ = "instant"
        self.compute(inplace=True)

    def __repr__(self):
        p = self.df_.shape[1] if self.df_.ndim > 1 else 1
        k = self.key_ if self.key_ is not None else "None"
        return "MetaPanda({}(n={}, p={}, mem={}, key='{}'), mode='{}')".format(
            self.name_,
            self.df_.shape[0],
            p,
            self.memory_,
            k,
            self.mode_
        )

    """ ############################### PROPERTIES ############################################## """

    """ DataFrame attributes """
    @property
    def df_(self):
        return self._df

    @df_.setter
    def df_(self, df):
        if isinstance(df, DataFrame):
            # apply the 'cleaning pipeline' to this.
            self._df = df
            # allocate meta data
            self._meta = basic_construct(self._df)
            # compute cleaning.
            self.compute(clean_pipe(), inplace=True)
            # add metadata columns
            add_metadata(self._df, self._meta)
            if "colnames" not in self._df.columns:
                self._df.columns.name = "colnames"
            if "counter" not in self._df.columns:
                self._df.index.name = "counter"
        elif isinstance(df, (pd.Series, DataFrameGroupBy)):
            # again, we'll just pretend the user knows what they're doing...
            self._df = df
        else:
            raise TypeError("'df' must be of type [pd.Series, pd.DataFrame, DataFrameGroupBy]")

    @property
    def meta_(self):
        return self._meta

    @meta_.setter
    def meta_(self, meta):
        if isinstance(meta, DataFrame):
            self._meta = meta
            # categorize
            categorize_meta(self._meta)
            # set colnames
            self._meta.index.name = "colnames"
        else:
            raise TypeError("'meta' must be of type [pd.DataFrame]")

    @property
    def n_(self):
        """
        Returns the number of rows within the df_ attribute.
        """
        return self.df_.shape[0]

    @property
    def p_(self):
        """
        Returns the number of columns/dimensions within the df_ attribute.
        """
        return self.df_.shape[1]

    """ Additional meta information """

    @property
    def selectors_(self):
        """ Returns the cached selectors available. """
        return self._select

    @property
    def memory_(self):
        return "{:0.3f}MB".format(calc_mem(self.df_) + calc_mem(self.meta_))

    @property
    def name_(self):
        return self._name

    @name_.setter
    def name_(self, n):
        if isinstance(n, str):
            # if name is found as a column name, block it.
            if n in self.df_.columns:
                raise ValueError("name: {} for df_ found as a column attribute; not allowed!".format(n))
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

    @property
    def pipe_(self):
        return self._pipe

    @pipe_.setter
    def pipe_(self, p):
        self._pipe = p

    @property
    def key_(self):
        return self._key

    @key_.setter
    def key_(self, k):
        if (k in self.df_.columns) and k in (self.view("is_unique") & self.view_not("is_missing")):
            self._key = k
        elif k is None:
            return
        else:
            raise ValueError("key '{}' belong in set:{}".format(k, k in self.df_.columns))

    """ ############################### BOOLEAN PROPERTIES ##################################################"""

    @property
    def is_square(self):
        return self.n_ == self.p_

    @property
    def is_symmetric(self):
        """ Defines whether a matrix is symmetric """
        return self.is_square and (np.allclose(self.df_, self.df_.T))

    @property
    def is_positive_definite(self):
        return np.all(np.linalg.eigvals(self.df_.values) > 0)

    @property
    def is_singular(self):
        return np.linalg.cond(self.df_.values) >= (1. / sys.float_info.epsilon)

    @property
    def is_orthogonal(self):
        """ An orthogonal matrix is square matrix whose columns are orthogonal unit vectors """
        return self.is_square and np.allclose(np.dot(self.df_.values, self.df_.values.T), np.eye(self.p_))

    """ ################################ PUBLIC FUNCTIONS ################################################### """

    def head(self, k=5):
        """
        A wrapper for pandas.DataFrame.head(). See pandas documentation for details.

        Parameters
        --------
        k : int
            Must be 0 < k < n.

        Returns
        -------
        ndf : pandas.DataFrame
            First k rows of df_
        """
        return self.df_.head(k)

    def view(self, *selector):
        """
        Select merely returns the columns of interest selected using this selector. This function
        is not affected by the 'mode' parameter. Note that view() *preserves* the order in which columns
        appear within the DataFrame.

        Selections of columns can be done by:
            type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
            callable (function) that returns [bool list] of length p
            pd.Index
            str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
            list/tuple of the above

        *: any numpy data type, like np.float64, np.uint8

        Parameters
        -------
        selector : str or tuple args
            See above for what constitutes an *appropriate selector*.

        Returns
        ------
        sel : list
            The list of column names selected, or empty
        """
        # we do this 'double-drop' to maintain the order of the dataframe, because of set operations.
        return self.df_.columns.drop(self.view_not(*selector))

    def view_not(self, *selector):
        """
        Select merely returns the columns of interest NOT selected using this selector. This function
        is not affected by the 'mode' parameter. Note that view_not() *preserves* the order in which columns
        appear within the DataFrame.

        Selections of columns can be done by:
            type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
            callable (function) that returns [bool list] of length p
            pd.Index
            str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
            list/tuple of the above

        *: any numpy data type, like np.float64, np.uint8

        Parameters
        -------
        selector : str or tuple args
            See above for what constitutes an *appropriate selector*.

        Returns
        ------
        sel : list
            The list of column names NOT selected, or empty
        """
        sel = get_selector(self.df_, self.meta_, self._select, selector, raise_error=False, select_join="OR")
        if (sel.shape[0] == 0) and self._with_warnings:
            warnings.warn("selection: '{}' was empty, no columns selected.".format(selector), UserWarning)
        # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
        return self.df_.columns.drop(sel)

    def copy(self):
        """
        Uses 'deepcopy' to create a copy of this object which is returned. This function
        is not affected by the 'mode' parameter.
        """
        return deepcopy(self)

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

    @_actionable
    def apply_columns(self, f_name, *f_args, **f_kwargs):
        """
        Applies a pandas.DataFrame.columns.* function to the MetaPanda columns. The result is then returned
        to the columns attribute, so it should only accept transform-like operations.

        e.g mdf.apply_columns("strip")
            applies self.df_.columns = self.df_.columns.str.strip() to columns

        Parameters
        -------
        f_name : str
            The name of the function. This can be in the .str accessor attribute also.
        f_args : list/tuple
            Arguments to pass to the function
        f_kwargs : dict
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_column_function(f_name, *f_args, **f_kwargs)
        return self

    @_actionable
    def apply_index(self, f_name, *f_args, **f_kwargs):
        """
        Applies a pandas.DataFrame.index.* function to the MetaPanda index. The result is then returned
        to the index attribute, so it should only accept transform-like operations.

        e.g mdf.apply_index("strip")
            applies self.df_.index = self.df_.index.str.strip() to columns

        Parameters
        -------
        f_name : str
            The name of the function. This can be in the .str accessor attribute also.
        f_args : list/tuple
            Arguments to pass to the function
        f_kwargs : dict
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_index_function(f_name, *f_args, **f_kwargs)
        return self

    @_actionable
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

    @_actionable
    def keep(self, *selector):
        """
        Given a selector or group of selectors, keep all of the columns selected within
        this group, applied to df_, dropping ALL others.

        Parameters
        --------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        -------
        self
        """
        self._drop_columns(self.view_not(*selector))
        return self

    @_actionable
    def filter_rows(self, function, selector=None, *args):
        """
        Given a function, filter out rows that do not meet the functions' criteria.

        Parameters
        --------
        function : function
            A function taking the whole dataset or subset, and returning a boolean
            pd.Series with True rows kept and False rows dropped
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
        if callable(function) and selection.shape[0] == 1:
            bs = function(self.df_[selection[0]], *args)
        elif callable(function) and selection.shape[0] > 1:
            bs = function(self.df_.loc[:, selection], *args)
        else:
            raise ValueError("parameter '{}' not callable".format(function))
        # check that bs is boolean series
        boolean_series_check(bs)
        self.df_ = self.df_.loc[bs, :]
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
        # convert selector over to list to make it mutable
        selector = list(selector)
        # encode to string
        enc_map = _type_encoder_map()
        # encode the selector as a string ALWAYS.
        for i, s in enumerate(selector):
            if s in enc_map:
                selector[i] = enc_map[s]
        # store to select
        self._select[name] = selector
        return self

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

    @_actionable
    def rename(self, ops, selector=None, axis=1):
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
        axis : int, optional
            0 for columns, 1 for index. default is 1 (for columns)

        Returns
        -------
        self or pd.Index
        """
        # check ops is right format
        is_twotuple(ops)
        belongs(axis, [0, 1])

        if selector is None:
            curr_cols = sel_cols = self.df_.columns if axis == 1 else self.df_.index
        elif axis==1:
            # ignore axis==0
            curr_cols = sel_cols = self.view(*selector)
        else:
            raise ValueError("cannot use argument [selector] with axis=0, for rows")

        for op in ops:
            curr_cols = curr_cols.str.replace(*op)
        self._rename_axis(sel_cols, curr_cols, axis)
        return self

    @_actionable
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

        Returns
        ------
        self
        """
        sel_cols = self.view(*selector) if selector is not None else self.df_.columns
        # set to df_ and meta_
        self._rename_axis(sel_cols, sel_cols + pref, 0)
        return self

    @_actionable
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

        Returns
        ------
        self
        """
        sel_cols = self.view(*selector) if selector is not None else self.df_.columns
        # set to df_ and meta_
        self._rename_axis(sel_cols, sel_cols + suf, 0)
        return self

    def analyze(self, functions=["agglomerate"]):
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

    @_actionable
    def transform(self, function, selector=None, whole=False, *args, **kwargs):
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
        whole : bool
            If True, uses function(self.df_, *args) and relies on the same output.
            If False, uses self.df_.transform(lambda x: function(x, *args, **kwargs))
        args : list
            Additional arguments to pass to function(x, *args)
        kwargs : dict
            Additional arguments to pass to function(x, *args, **kwargs)

        Returns
        -------
        self
        """
        # perform inplace
        if isinstance(selector, (tuple, list)):
            selection = self.view(*selector)
        else:
            selection = self.view(selector) if selector is not None else self.df_.columns
        # modify
        if callable(function) and selection.shape[0] > 0:
            if whole:
                self.df_.loc[:, selection] = function(self.df_.loc[:, selection], *args, **kwargs)
            else:
                self.df_.loc[:, selection] = self.df_.loc[:, selection].transform(
                    lambda x: function(x, *args, **kwargs))
        return self

    @_actionable
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

    @_actionable
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
            names = [sel if sel in self._select else "group%d" % (i + 1) for i, sel in enumerate(selectors)]
        else:
            raise TypeError("'selectors' must be of type [list, tuple, dict]")

        igrid = intersection_grid(cnames)
        if igrid.shape[0] == 0:
            new_grid = pd.concat([pd.Series(n, index=val) for n, val in zip(names, cnames)], sort=False, axis=0)
            new_grid.name = name
        else:
            raise ValueError("shared terms: {} discovered for meta_map.".format(igrid))
        # merge into meta
        self.meta_[name] = object_to_categorical(new_grid)
        return self

    @_actionable
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
                self._df = self.df_.reindex(self.meta_.index, axis=1)
        elif isinstance(by, (list, tuple)):
            if "alphabet" in by:
                by[by.index("alphabet")] = "colnames"
            if isinstance(ascending, bool):
                # turn into a list with that value
                ascending = [ascending] * len(by)
            if len(by) != len(ascending):
                raise ValueError(
                    "the length of 'by' {} must equal the length of 'ascending' {}".format(len(by), len(ascending)))
            if all([(col in self.meta_) or (col == "colnames") for col in by]):
                # sort meta
                self.meta_.sort_values(by=by, axis=0, ascending=ascending, inplace=True)
                # sort df
                self._df = self.df_.reindex(self.meta_.index, axis=1)
        else:
            raise TypeError("'by' must be of type [str, list, tuple], not {}".format(type(by)))
        return self

    @_actionable
    def expand(self, column, sep=","):
        """
        Expands out a 'stacked' id column to a longer-form DataFrame, and re-merging
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

    @_actionable
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
        if (column not in self.df_.columns) and (column != self.df_.index.name):
            raise ValueError("column '{}' not found in df".format(column))

        # no changes made to columns, use hidden df
        self._df = pd.merge(
            # shrink down id column
            self.df_.groupby("counter")[column].apply(lambda x: x.str.cat(sep=sep)),
            self.df_.reset_index().drop_duplicates("counter").set_index("counter").drop(column, axis=1),
            left_index=True, right_index=True
        )
        self._df.columns.name = "colnames"
        return self

    @_actionable
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
            cnames = ["cat%d" % (i + 1) for i in range(exp.shape[1])]
        else:
            cnames = renames if len(renames) == exp.shape[1] else ["cat%d" % (i + 1) for i in range(exp.shape[1])]

        self._df = self.df_.join(
            exp.rename(columns=dict(zip(range(exp.shape[1]), cnames)))
        )
        self._df.columns.name = "colnames"
        return self

    @_actionable
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
            self.df_ = self.df_.reset_index().melt(id_vars=id_vars, *args, **kwargs)
        return self

    def compute(self, pipe=None, inplace=False):
        """
        Computes a pipeline to the MetaPanda object. If there are no parameters, it computes
        what is stored in the pipe_ attribute, if any.

        Parameters
        -------
        pipe : list of 3-tuple, (function name, *args, **kwargs), multiple pipes, optional
            A set of instructions expecting function names in MetaPanda and parameters.
            If empty, computes the stored pipe_ attribute.
        inplace : bool, optional
            If True, applies the pipe inplace, else returns a copy. Default has now changed
            to return a copy.

        Returns
        -------
        self/copy
        """
        if pipe is None:
            # use self.pipe_
            pipe = self.pipe_
            self.pipe_ = []

        if inplace:
            # computes inplace
            self.mode_ = "instant"
            self._apply_pipe(pipe)
            return self
        else:
            # full deepcopy, including dicts, lists, hidden etc.
            cpy = self.copy()
            # compute on cop
            cpy.compute(pipe, inplace=True)
            return cpy

    def multi_compute(self, pipes, inplace=False):
        """
        Computes multiple pipeslines to the MetaPanda object. This version does not compute on the pipe_ attribute.
        By default, this method does NOT

        Parameters
        --------
        pipes : list of (list of (list of 3-tuple, (function name, *args, **kwargs))
            A set of instructions expecting function names in MetaPanda and parameters.
            If empty, computes nothing.
        inplace : bool, optional
            If True, applies the pipes inplace, else returns a copy. Default has now changed
            to return a copy.

        Returns
        -------
        self/copy
        """
        # join and execute
        return self.compute(join(*pipes), inplace=inplace)

    def write(self, filename=None, with_meta=False, *args, **kwargs):
        """
        Saves a MetaPanda to disk.

        This function is not affected by the 'mode' parameter.

        Parameters
        -------
        filename : str, optional
            The name of the file to save, or None it will create a JSON file with Data.
            Accepts filename that end in [.csv, .json]
        with_meta : bool, optional
            If true, saves metafile also, else doesn't
        *args : list
            Arguments to pass to pandas.to_csv, not used with JSON file
        **kwargs : dict
            Keywords to pass to pandas.to_csv, not used with JSON file

        Returns
        -------
        None
        """
        if filename is None:
            self._write_json(self.name_ + ".json")
        elif filename.endswith(".csv"):
            self._write_csv(filename, with_meta, *args, **kwargs)
        elif filename.endswith(".json"):
            self._write_json(filename)
        else:
            raise IOError("Doesn't recognize filename or type: '{}', must end in [csv, json]".format(filename))

    _actionable = staticmethod(_actionable)
