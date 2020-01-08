#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
"""This file handles the use of MetaPanda objects."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import json
import sys
import warnings
from copy import deepcopy
from functools import wraps

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

# locals
from .analyze import intersection_grid
from .metadata import *
from .pipe import Pipe
from .selection import get_selector, _type_encoder_map
from .utils import *

__all__ = ("MetaPanda")


class MetaPanda(object):
    """A meta-object for a pandas.DataFrame with extra information.

    The aim of this object is to extend the functionality of the `pandas` library package,
    which is extensively used for data munging, manipulation and visualization of large datasets.

    Attributes
    ----------
    df_ : pd.DataFrame
        The underlying (n, p)-shaped dataset
    meta_ : pd.DataFrame
        Meta information (p, k)-shaped describing column data
    name_ : str
        The name of the dataset
    n_ : int
        The number of rows in `df_`
    p_ : int
        The number of columns in `df_`
    memory_ : str
        String-formatted representation of memory consumption in megabytes
    mode_ : str
        Determines how 'delay' methods are executed
    selectors_ : dict
        Maps unique name (key) to cached selected groups of columns (value)
    mapper_ : dict
        Maps unique name (key) to key of cached selected column groups (value)
    pipe_ : dict
        Maps unique name (key) to cached Pipe objects

    Methods
    -------
    head(k=5)
         A wrapper for pandas.DataFrame.head()
    dtypes(grouped=True)
        Determines the data type of each column in `df_`
    view(selector)
        Views a selection of columns in `df_`
    search(selector)
        View the intersection of search terms, for columns in `df_`.
    view_not(selector)
        Views the non-selected columns in `df_`
    copy(None)
        Creates a copy of this instance
    apply(f_name, f_args, f_kwargs)
        Applies a pandas.DataFrame function to `df_`
    apply_index(f_name, f_args, f_kwargs)
        Applies a pandas.Index function to `df_.index`
    apply_columns(f_name, f_args, f_kwargs)
        Applies a pandas.Index function to `df_.columns`
    drop(selector)
        Drops the selected columns from `df_`
    keep(selector)
        Keeps the selected columns from `df_` only
    filter_rows(func, selector=None, args)
        Filters j rows using boolean-index returned from `function`
    cache(name, selector)
        Adds a cache element to `selectors_`
    cache_k(selectors)
        Adds k cache elements to `selectors_`
    cache_pipe(name, pipe)
        Adds a pipe element to `pipe_`
    rename(ops, selector=None, axis=1)
        Performs a chain of .str.replace operations on `df_.columns`
    add_prefix(pref, selector=None)
        Adds a string prefix to selected columns
    add_suffix(suf, selector=None)
        Adds a string suffix to selected columns
    transform(func, selector=None,  method='transform', whole=False, args, kwargs)
        Performs an inplace transformation to a group of columns within `df_`.
    transform_k(ops)
        Performs multiple inplace transformations to a group of columns within `df_`
    meta_map(name, selectors)
        Maps a group of selectors with an identifier, in `mapper_`
    sort_columns(by="alphabet", ascending=True)
        Sorts `df_` using vast selection criteria
    expand(column, sep=",")
        Expands out a 'stacked' id column to a longer-form DataFrame
    shrink(column, sep=",")
        Expands out a 'unstacked' id column to a shorter-form DataFrame
    split_categories(column, sep=",", renames=None)
        Splits a column into j categorical variables to be associated with `df_`
    compute(pipe=None, inplace=False, update_meta=False)
        Executes a pipeline on `df_`
    compute_k(pipes=None, inplace=False)
        Executes `k` pipelines on `df_`, in order
    write(filename=None)
        Saves a MetaPanda to disk
    """

    def __init__(self,
                 dataset,
                 name: str = None,
                 mode: str = "instant",
                 with_clean: bool = True,
                 with_warnings: bool = False):
        """Define a MetaPanda frame.

        Creates a Meta DataFrame with the raw data and parametrization of
        the DataFrame by its grouped columns.

        Parameters
        ----------
        dataset : pd.DataFrame
            The raw data set to create as a MetaDataFrame.
        name : str, optional
            Gives the MetaDataFrame a name, which comes into play with merging, for instance
        mode : str, optional
            Choose from {'instant', 'delay'}
            If 'instant', executes all functions immediately inplace
            If 'delay', builds a task graph in 'pipe_'
                and then executes inplace when 'compute' is called
        with_clean : bool, optional
            If True, uses Pipe.clean() to perform minor preprocessing on `dataset`
        with_warnings : bool, optional
            If True, prints warnings when strange things happen, for instance during selection
        """
        self._with_warnings = with_warnings
        self._with_clean = with_clean

        self.mode_ = mode
        # selectors saved
        self._select = {}
        # pipeline arguments
        self._pipe = {"current": []}
        # columns added to meta_map
        self._mapper = {}
        # set empty meta
        self._meta = None
        # set using property
        self.df_ = dataset
        self.name_ = name if name is not None else 'DataSet'

    """ ############################ STATIC FUNCTIONS ######################################## """

    def _actionable(function):
        @wraps(function)
        def new_function(self, *args, **kwargs):
            if self.mode_ == "delay":
                self._pipe["current"].append((function.__name__, args, kwargs))
            else:
                return function(self, *args, **kwargs)

        return new_function

    @classmethod
    def from_pandas(cls, filename, name=None, *args, **kwargs):
        """Read in a MetaPanda from a comma-separated version (csv) file.

        Parameters
        -------
        filename : str
            A relative/absolute link to the file, with extension provided.
            Accepted extensions: {'csv', 'xls', 'xlsx', 'XLSX', 'hdf', 'sql'}
        name : str, optional
            A custom name to use for the MetaPanda.
            If None, `filename` is used.
        args : list, optional
            Additional args to pass to pd.read_csv
        kwargs : dict, optional
            Additional args to pass to pd.read_csv

        Returns
        ------
        mdf : MetaPanda
            A MetaPanda object.
        """
        file_ext_map = {
            "csv": pd.read_csv, "xls": pd.read_excel, "xlsx": pd.read_excel,
            "hdf": pd.read_hdf, "sql": pd.read_sql, "XLSX": pd.read_excel
        }

        # split filename into parts and check
        directory, jname, ext = split_file_directory(filename)

        df = file_ext_map[ext](filename, *args, **kwargs)
        # create MetaPanda
        mp = cls(df, name=name) if name is not None else cls(df, name=jname)
        # return
        return mp

    @classmethod
    def from_json(cls, filename, **kwargs):
        """Read in a MetaPanda from a custom JSON file.

        Reads in a datafile from JSON and creates a MetaPanda object from it.
        Pipe attributes are not saved currently due to the problem of storing
        potential lambda functions.

        Parameters
        -------
        filename : str
            A relative/absolute link to the JSON file.
        **kwargs : dict, optional
            Other keyword arguments to pass to MetaPanda constructor

        Returns
        ------
        mpf : MetaPanda
            A MetaPanda object.
        """
        # read in JSON
        with open(filename, "r") as f:
            mp = json.load(f)
        # go over attributes and assign where available
        if "data" in mp.keys():
            df = pd.DataFrame.from_dict(mp["data"])
            df.index.name = "counter"
            df.columns.name = "colnames"
            # assign to self
            mpf = cls(df, **kwargs)
        else:
            raise ValueError("column 'data' not found in MetaPandaJSON")

        if "cache" in mp.keys():
            mpf._select = mp["cache"]
        if "mapper" in mp.keys():
            mpf._mapper = mp["mapper"]
        if "name" in mp.keys() and "name" not in kwargs.keys():
            mpf.name_ = mp["name"]
        if "pipe" in mp.keys():
            mpf._pipe = mp["pipe"]

        # define metamaps if they exist
        mpf._define_metamaps()

        return mpf

    """ ############################ HIDDEN OPERATIONS ####################################### """

    def _selector_group(self, s, axis=1):
        if s is None:
            return self.df_.columns if axis == 1 else self.df_.index
        elif axis == 1:
            if isinstance(s, (tuple, list)):
                return self.view(*s)
            else:
                return self.view(s)
        else:
            raise ValueError("cannot use argument [selector] with axis=0, for rows")

    def _rename_axis(self, old, new, axis=1):
        if axis == 1:
            self.df_.rename(columns=dict(zip(old, new)), inplace=True)
            self.meta_.rename(index=dict(zip(old, new)), inplace=True)
        elif axis == 0:
            self.df_.rename(index=dict(zip(old, new)), inplace=True)
        else:
            raise ValueError("axis '{}' not recognized".format(axis))

    def _remove_unused_categories(self):
        for col in self.meta_.columns[self.meta_.dtypes == "category"]:
            self.meta_[col].cat.remove_unused_categories(inplace=True)

    def _drop_columns(self, select):
        if select.size > 0:
            self.df_.drop(select, axis=1, inplace=True)
            self.meta_.drop(select, axis=0, inplace=True)
            # remove any unused categories that might've been dropped
            self._remove_unused_categories()

    def _apply_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_, fn):
            self._df = getattr(self.df_, fn)(*fargs, **fkwargs)
            return self
        # if we start with groupby, then we are coupling groupby with an aggregation.
        elif fn.startswith("groupby__"):
            _, fn2 = fn.split("__", 1)
            _grouped = getattr(self.df_, "groupby")(*fargs, **fkwargs)
            if hasattr(_grouped, fn2):
                self._df = _grouped.agg(fn2)
                return self
            else:
                raise ValueError(
                    "function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn2, dir(_grouped)))
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.* API: {}".format(fn, dir(self.df_)))

    def _apply_index_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_.index, fn):
            self.df_.index = getattr(self.df_.index, fn)(*fargs, **fkwargs)
            return self
        elif hasattr(self.df_.index.str, fn):
            self.df_.index = getattr(self.df_.index.str, fn)(*fargs, **fkwargs)
            return self
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.index.[str.]* API".format(fn))

    def _apply_column_function(self, fn, *fargs, **fkwargs):
        if hasattr(self.df_.columns, fn):
            self.df_.columns = getattr(self.df_.columns, fn)(*fargs, **fkwargs)
            self.meta_.index = getattr(self.meta_.index, fn)(*fargs, **fkwargs)
            return self
        elif hasattr(self.df_.columns.str, fn):
            self.df_.columns = getattr(self.df_.columns.str, fn)(*fargs, **fkwargs)
            self.meta_.index = getattr(self.meta_.index.str, fn)(*fargs, **fkwargs)
            return self
        else:
            raise ValueError("function '{}' not recognized in pandas.DataFrame.columns.[str.]* API".format(fn))

    def _reset_meta(self):
        self._meta = basic_construct(self._df)
        # add in metadata rows.
        add_metadata(self._df, self._meta)
        # if we have mapper elements, add these in
        if len(self._mapper) > 0:
            self._define_metamaps()

    def _define_metamaps(self):
        if len(self.mapper_) > 0:
            for k, v in self.mapper_.items():
                self.meta_map(k, v)

    def _apply_pipe(self, pipe):
        # checks
        if isinstance(pipe, str):
            # see if name is cached away.
            if pipe in self.pipe_.keys():
                pipe = self.pipe_[pipe]
            else:
                raise ValueError("pipe name '{}' not found in .pipe attribute".format(pipe))
        elif isinstance(pipe, Pipe):
            pipe = pipe.p
        if isinstance(pipe, (list, tuple)):
            if len(pipe) == 0 and self._with_warnings:
                warnings.warn("pipe_ element empty, nothing to compute.", UserWarning)
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
            directory, jname, ext = split_file_directory(filename)
            self.meta_.to_csv(directory + "/" + jname + "__meta.csv", sep=",")

    def _write_json(self, filename):
        # columns founded by meta_map are dropped
        redundant_meta = meta_columns_default() + list(self.mapper_.keys())
        # saving_dict
        sd = {
            "data": self.df_.to_dict(),
            "name": self.name_
        }
        # only write these if we have data to write - this saves space.
        reduced_meta = self.meta_.drop(redundant_meta, axis=1).to_dict()
        if len(reduced_meta) > 0:
            sd['meta'] = reduced_meta
        if len(self.selectors_) > 0:
            sd['cache'] = self._select
        if len(self.pipe_) > 0:
            sd['pipe'] = self.pipe_
        if len(self.mapper_) > 0:
            sd['mapper'] = self.mapper_

        fn = filename if filename is not None else self.name_ + '.json'
        with open(fn, "w") as f:
            json.dump(sd, f, separators=(",", ":"))

    """ ############################## OVERRIDDEN OPERATIONS ###################################### """

    def __getitem__(self, selector):
        """Fetch a subset determined by the selector."""
        # we take the columns that are NOT this selection, then drop to keep order.
        sel = self._selector_group(selector, axis=1)
        if sel.size > 0:
            # drop anti-selection to maintain order/sorting
            return self.df_[sel].squeeze()

    def __delitem__(self, selector):
        """Delete columns determined by the selector."""
        # drops columns inplace
        self._drop_columns(self.view(selector))

    def __repr__(self):
        """Represent the object in a stringed format."""
        p = self.df_.shape[1] if self.df_.ndim > 1 else 1
        """
        OPTIONS are:
        - S: contains selectors
        - P: contains pipe attributes
        - M: contains mapper(s)
        """
        opt_s = "S" if len(self.selectors_) > 0 else ''
        opt_p = "P" if len(self.pipe_) > 1 else ''
        opt_m = "M" if len(self.mapper_) > 0 else ''
        opts = "[" + opt_s + opt_p + opt_m + ']'
        return "MetaPanda({}(n={}, p={}, mem={}, options={}))".format(
            self.name_,
            self.df_.shape[0],
            p,
            self.memory_,
            opts
        )

    """ ############################### PROPERTIES ############################################## """

    """ DataFrame attributes """

    @property
    def df_(self):
        """Fetch the raw dataset."""
        return self._df

    @df_.setter
    def df_(self, df):
        if isinstance(df, pd.DataFrame):
            # apply the 'cleaning pipeline' to this.
            self._df = df
            # define meta
            self._reset_meta()
            # compute cleaning.
            if self._with_clean:
                self.compute(Pipe.clean(), inplace=True)

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
        """Fetch the dataframe with meta information."""
        return self._meta

    @meta_.setter
    def meta_(self, meta):
        if isinstance(meta, pd.DataFrame):
            self._meta = meta
            # categorize
            categorize_meta(self._meta)
            # set colnames
            self._meta.index.name = "colnames"
        else:
            raise TypeError("'meta' must be of type [pd.DataFrame]")

    @property
    def n_(self):
        """Fetch the number of rows/samples within the df_ attribute."""
        return self.df_.shape[0]

    @property
    def p_(self):
        """Fetch the number of dimensions within the df_ attribute."""
        return self.df_.shape[1]

    """ Additional meta information """

    @property
    def selectors_(self):
        """Fetch the cached selectors available."""
        return self._select

    @property
    def memory_(self):
        """Fetch the memory consumption of the MetaPanda."""
        return "{:0.3f}MB".format(calc_mem(self.df_) + calc_mem(self.meta_))

    @property
    def name_(self):
        """Fetch the name of the MetaPanda."""
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
        """Choose from {'instant', 'delay'}."""
        return self._mode

    @mode_.setter
    def mode_(self, mode):
        if mode in ("instant", "delay"):
            self._mode = mode
        else:
            raise ValueError("'mode' must be ['instant', 'delay'], not '{}'".format(mode))

    @property
    def pipe_(self):
        """Fetch the cached pipelines."""
        return self._pipe

    @property
    def mapper_(self):
        """Fetch the mapping between unique name and selector groups."""
        return self._mapper

    """ ############################### BOOLEAN PROPERTIES ##################################################"""

    @property
    def is_square(self):
        """Determine whether the matrix is square-shaped."""
        return self.n_ == self.p_

    @property
    def is_symmetric(self):
        """Determine whether the matrix is symmetric."""
        return self.is_square and (np.allclose(self.df_, self.df_.T))

    @property
    def is_positive_definite(self):
        """Determine whether the matrix is positive-definite."""
        return self.is_square and np.all(np.linalg.eigvals(self.df_.values) > 0)

    @property
    def is_singular(self):
        """Determine whether the matrix is singular."""
        return np.linalg.cond(self.df_.values) >= (1. / sys.float_info.epsilon)

    @property
    def is_orthogonal(self):
        """Determine whether the matrix is orthogonal."""
        return self.is_square and np.allclose(np.dot(self.df_.values, self.df_.values.T), np.eye(self.p_))

    """ ################################ PUBLIC FUNCTIONS ################################################### """

    def head(self, k: int = 5):
        """Look at the top k rows of the dataset.

        See `pd.DataFrame.head` documentation for details.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        --------
        k : int, optional
            Must be 0 < k < n.

        Returns
        -------
        ndf : pandas.DataFrame
            First k rows of df_
        """
        return self.df_.head(k)

    def dtypes(self, grouped: bool = True):
        """Determine the grouped data types in the dataset.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        --------
        grouped : bool, optional
            If True, returns the value_counts of each data type, else returns the direct types.

        Returns
        -------
        true_types : pd.Series
            A series of index (group/name) and value (count/type)
        """
        return self.meta_['e_types'].value_counts() if grouped else self.meta_['e_types']

    def view(self, *selector):
        """View a selection of columns in `df_`.

        Select merely returns the columns of interest selected using this selector.
        Selections of columns can be done by:
            type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
            callable (function) that returns [bool list] of length p
            pd.Index
            str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
            list/tuple of the above

        .. warning:: Not affected by `mode_` attribute.

        .. note:: `view` *preserves* the order in which columns appear within the DataFrame.

        .. note:: any numpy data type, like np.float64, np.uint8

        Parameters
        -------
        selector : str or tuple args
            See above for what constitutes an *appropriate selector*.

        Warnings
        --------
        UserWarning
            If the selection returned is empty.

        Returns
        ------
        sel : list
            The list of column names selected, or empty

        See Also
        --------
        view_not : View the non-selected columns in `df_`.
        search : View the intersection of search terms, for columns in `df_`.
        """
        # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
        return self.df_.columns.drop(self.view_not(*selector))

    def search(self, *selector):
        """View the intersection of search terms, for columns in `df_`.

        Select merely returns the columns of interest selected using this selector.
        Selections of columns can be done by:
            type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
            callable (function) that returns [bool list] of length p
            pd.Index
            str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
            list/tuple of the above

        .. warning:: Not affected by `mode_` attribute.

        .. note:: `view` *preserves* the order in which columns appear within the DataFrame.

        .. note:: any numpy data type, like np.float64, np.uint8

        Parameters
        -------
        selector : str or tuple args
            See above for what constitutes an *appropriate selector*.

        Warnings
        --------
        UserWarning
            If the selection returned is empty.

        Returns
        ------
        sel : list
            The list of column names selected, or empty

        See Also
        --------
        view_not : Views the non-selected columns in `df_`.
        view : View a selection of columns in `df_`.
        """
        # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
        sel = get_selector(self.df_, self.meta_, self._select, selector, raise_error=False, select_join="AND")
        if (sel.shape[0] == 0) and self._with_warnings:
            warnings.warn("in select: '{}' was empty, no columns selected.".format(selector), UserWarning)
        # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
        return sel

    def view_not(self, *selector):
        """View the non-selected columns in `df_`.

        Select merely returns the columns of interest NOT selected using this selector.
        Selections of columns can be done by:
            type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
            callable (function) that returns [bool list] of length p
            pd.Index
            str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
            list/tuple of the above

        .. warning:: Not affected by `mode_` attribute.

        .. note:: `view_not` *preserves* the order in which columns appear within the DataFrame.

        .. note::  any numpy data type, like np.float64, np.uint8.

        Parameters
        -------
        selector : str or tuple args
            See above for what constitutes an *appropriate selector*.

        Warnings
        --------
        UserWarning
            If the selection returned is empty.

        Returns
        ------
        sel : list
            The list of column names NOT selected, or empty

        See Also
        --------
        view : View a selection of columns in `df_`.
        search : View the intersection of search terms, for columns in `df_`.
        """
        sel = get_selector(self.df_, self.meta_, self._select, selector, raise_error=False, select_join="OR")
        if (sel.shape[0] == 0) and self._with_warnings:
            warnings.warn("in view: '{}' was empty, no columns selected.".format(selector), UserWarning)
        # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
        return self.df_.columns.drop(sel)

    def copy(self):
        """Create a copy of this instance.

        .. warning:: Not affected by `mode_` attribute.

        Raises
        ------
        CopyException
            Module specific errors with copy.deepcopy

        Returns
        -------
        mdf2 : MetaPanda
            A copy of this object

        See Also
        --------
        copy.deepcopy(x) : Return a deep copy of x.
        """
        return deepcopy(self)

    @_actionable
    def apply(self, f_name: str, *f_args, **f_kwargs):
        """Apply a `pd.DataFrame` function to `df_`.

        e.g mdf.apply("groupby", ["counter","refseq_id"], as_index=False)
            applies self.df_.groupby() to data and return value is stored in df_
            assumes pandas.DataFrame is returned.

        Parameters
        -------
        f_name : str
            The name of the function
        f_args : list/tuple, optional
            Arguments to pass to the function
        f_kwargs : dict, optional
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_function(f_name, *f_args, **f_kwargs)
        return self

    @_actionable
    def apply_columns(self, f_name: str, *f_args, **f_kwargs):
        """Apply a `pd.Index` function to `df_.columns`.

        The result is then returned to the columns attribute, so it should only accept transform-like operations.

        Thus to apply `strip` to all column names:

        >>> mdf.apply_columns("strip")

        Parameters
        -------
        f_name : str
            The name of the function. This can be in the .str accessor attribute also.
        f_args : list/tuple, optional
            Arguments to pass to the function
        f_kwargs : dict, optional
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_column_function(f_name, *f_args, **f_kwargs)
        return self

    @_actionable
    def apply_index(self, f_name: str, *f_args, **f_kwargs):
        """Apply a `pd.Index` function to `df_.index`.

        The result is then returned to the index attribute, so it should only accept transform-like operations.

        Thus to apply `strip` to all index names:

        >>> mdf.apply_columns("strip")

        Parameters
        -------
        f_name : str
            The name of the function. This can be in the .str accessor attribute also.
        f_args : list/tuple, optional
            Arguments to pass to the function
        f_kwargs : dict, optional
            Keyword arguments to pass to the function

        Returns
        -------
        self
        """
        self._apply_index_function(f_name, *f_args, **f_kwargs)
        return self

    @_actionable
    def drop(self, *selector):
        """Drop the selected columns from `df_`.

        Given a selector or group of selectors, drop all of the columns selected within
        this group, applied to `df_`.

        .. note:: `drop` *preserves* the order in which columns appear within the DataFrame.

        Parameters
        -------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        -------
        self

        See Also
        --------
        keep : Keeps the selected columns from `df_` only.
        """
        # perform inplace
        self._drop_columns(self.view(*selector))
        return self

    @_actionable
    def keep(self, *selector):
        """Keep the selected columns from `df_` only.

        Given a selector or group of selectors, keep all of the columns selected within
        this group, applied to `df_`, dropping all others.

        .. note:: `keep` *preserves* the order in which columns appear within the DataFrame.

        Parameters
        --------
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Returns
        -------
        self

        See Also
        --------
        drop : Drops the selected columns from `df_`.
        """
        self._drop_columns(self.view_not(*selector))
        return self

    @_actionable
    def filter_rows(self, func: callable, selector=None, *args):
        """Filter j rows using boolean-index returned from `function`.

        Given a function, filter out rows that do not meet the functions' criteria.

        .. note:: if `selector` is set, the filtering only factors in these columns.

        Parameters
        --------
        func : function
            A function taking the whole dataset or subset, and returning a boolean
            `pd.Series` with True rows kept and False rows dropped
        selector : str or tuple args, optional
            Contains either types, meta column names, column names or regex-compliant strings.
            If None, applies `func` to all columns.
        args : list, optional
            Additional arguments to pass as `func(x, *args)`

        Returns
        -------
        self
        """
        # perform inplace
        selection = self._selector_group(selector, axis=1)
        # modify
        if callable(func) and selection.shape[0] == 1:
            bs = func(self.df_[selection[0]], *args)
        elif callable(func) and selection.shape[0] > 1:
            bs = func(self.df_.loc[:, selection], *args)
        else:
            raise ValueError("parameter '{}' not callable".format(func))
        # check that bs is boolean series
        boolean_series_check(bs)
        self.df_ = self.df_.loc[bs, :]
        return self

    def cache(self, name: str, *selector):
        """Add a cache element to `selectors_`.

        Saves a 'selector' to use at a later date. This can be useful if you
        wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        -------
        name : str
            A name to reference the selector with.
        selector : str or tuple args
            Contains either types, meta column names, column names or regex-compliant strings

        Warnings
        --------
        UserWarning
            Raised if `name` already exists in `selectors_`, overrides by default.

        Returns
        -------
        self

        See Also
        --------
        cache_k : Adds k cache elements to `selectors_.
        cache_pipe : Adds a pipe element to `pipe_`.
        """
        if name in self._select and self._with_warnings:
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

    def cache_k(self, **caches):
        """Add k cache elements to `selectors_`.

        Saves a group of 'selectors' to use at a later date. This can be useful
        if you wish to keep track of changes, or if you want to quickly reference a selector
        using a name rather than a group of selections.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        --------
        caches : dict (k, w)
            keyword: unique reference of the selector
            value: selector: str, tuple args
                 Contains either types, meta column names, column names or regex-compliant

        Warnings
        --------
        UserWarning
            Raised if one of keys already exists in `selectors_`, overrides by default.

        Returns
        -------
        self

        See Also
        --------
        cache : Adds a cache element to `selectors_`.
        cache_pipe : Adds a pipe element to `pipe_`.
        """
        for name, selector in caches.items():
            if isinstance(selector, (tuple, list)) and len(selector) > 1:
                self.cache(name, *selector)
            else:
                self.cache(name, selector)
        return self

    def cache_pipe(self, name: str, pipeline):
        """Add a pipe element to `pipe_`.

        Saves a pipeline to use at a later date. Calls to `compute` can reference the name
        of the pipeline.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        ----------
        name : str
            A name to reference the pipeline with.
        pipeline : Pipe, list, tuple
            list of 3-tuple, (function name, *args, **kwargs), multiple pipes, optional
            A set of instructions expecting function names in MetaPanda and parameters.
            If None, computes the stored `pipe_` attribute.

        Warnings
        --------
        UserWarning
            Raised if `name` already exists in `pipe_`, overrides by default.

        Returns
        -------
        self
        """
        if name in self.pipe_.keys() and self._with_warnings:
            warnings.warn("pipe name '{}' already exists in .pipe, overriding".format(name), UserWarning)
        if isinstance(pipeline, Pipe):
            pipeline = pipeline.p
        self.pipe_[name] = pipeline
        return self

    @_actionable
    def rename(self, ops, selector=None, axis: int = 1):
        """Perform a chain of .str.replace operations on `df_.columns`.

        .. note:: strings that are unchanged remain the same (are not NA'd).

        Parameters
        -------
        ops : list of tuple (2,)
            Where the first value of each tuple is the string to find, with its replacement
            At this stage we only accept *direct* replacements. No regex.
            Operations are performed 'in order'.
        selector : None, str, or tuple args, optional
            Contains either types, meta column names, column names or regex-compliant strings
            If None, all column names are subject to potential renaming
        axis : int, optional
            Choose from {1, 0} 1 = columns, 0 = index.

        Returns
        -------
        self
        """
        # check ops is right format
        is_twotuple(ops)
        belongs(axis, [0, 1])

        curr_cols = sel_cols = self._selector_group(selector, axis)
        # performs the replacement operation inplace
        curr_cols = string_replace(curr_cols, ops)
        # rename using mapping
        self._rename_axis(sel_cols, curr_cols, axis)
        return self

    @_actionable
    def add_prefix(self, pref: str, selector=None):
        """Add a prefix to all of the columns or selected columns.

        Parameters
        -------
        pref : str
            The prefix to add
        selector : None, str, or tuple args, optional
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename

        Returns
        ------
        self
        """
        sel_cols = self._selector_group(selector)
        # set to df_ and meta_
        self._rename_axis(sel_cols, sel_cols + pref, axis=1)
        return self

    @_actionable
    def add_suffix(self, suf: str, selector=None):
        """Add a suffix to all of the columns or selected columns.

        Parameters
        -------
        suf : str
            The prefix to add
        selector : None, str, or tuple args, optional
            Contains either types, meta column names, column names or regex-compliant strings
            Allows user to specify subset to rename

        Returns
        ------
        self
        """
        sel_cols = self._selector_group(selector)
        # set to df_ and meta_
        self._rename_axis(sel_cols, sel_cols + suf, axis=1)
        return self

    @_actionable
    def transform(self,
                  func: callable,
                  selector=None,
                  method: str = 'transform',
                  whole: bool = False,
                  *args,
                  **kwargs):
        """Perform an inplace transformation to a group of columns within the `df_` attribute.

        This flexible function provides capacity for a wide-range of transformations, including custom transformations.

        .. note:: `func` must be a transforming function, i.e one that does not change the shape of `df_`.

        Parameters
        -------
        func : function
            A function taking the `pd.Series` x as input and returning `pd.Series` y as output
            If `whole`, accepts `pd.DataFrame` X, returning `pd.DataFrame` Y
        selector : None, str, or tuple args, optional
            Contains either types, meta column names, column names or regex-compliant strings
            If None, applies the function to all columns.
        method : str, optional
            Allows the user to specify which underlying DataFrame function to call.
                Choose from {'transform', 'apply', 'applymap'}
                - 'transform': Provides shape guarantees and computationally cheapest.
                - 'apply': more generic function. Can be expensive.
                - 'applymap': applies to every ELEMENT (not axis).
                See `pd.DataFrame` for more details.
        whole : bool, optional
            If True, applies whole function. Often computationally cheaper.
            If False, makes use of `pd.DataFrame.<method>`, see `method` parameter. More flexible.
        args : list, optional
            Additional arguments to pass to function(x, *args)
        kwargs : dict, optional
            Additional arguments to pass to function(x, *args, **kwargs)

        Returns
        -------
        self

        See Also
        --------
        transform_k : Performs multiple inplace transformations to a group of columns within `df_`.
        """
        belongs(method, ['apply', 'transform', 'applymap'])
        # perform inplace
        selection = self._selector_group(selector)
        # modify
        if callable(func) and selection.shape[0] > 0:
            if whole:
                self.df_.loc[:, selection] = func(self.df_.loc[:, selection], *args, **kwargs)
            else:
                self.df_.loc[:, selection] = getattr(self.df_.loc[:, selection], method)(func, *args, **kwargs)
        return self

    @_actionable
    def transform_k(self, ops):
        """Perform multiple inplace transformations to a group of columns within `df_`.

        Allows a chain of k transformations to be applied, in order.

        Parameters
        -------
        ops : list of 2-tuple
            Containing:
                1. func : A function taking the pd.Series x_i as input and returning pd.Series y_i as output
                2. selector : Contains either types, meta column names, column names or regex-compliant strings
                    Allows user to specify subset to rename

        Raises
        ------
        ValueException
            ops must be a 2-tuple shape

        Returns
        -------
        self

        See Also
        --------
        transform : Performs an inplace transformation to a group of columns within the `df_` attribute.
        """
        is_twotuple(ops)
        for op in ops:
            self.transform(op[0], op[1])
        return self

    @_actionable
    def meta_map(self, name: str, selectors):
        """Map a group of selectors with an identifier, in `mapper_`.

        Maps a group of selectors into a column in the meta-information
        describing some groupings/associations between features. For example,
        your data may come from multiple sources and you wish to
        identify this within your data.

        Parameters
        --------
        name : str
            The name of this overall grouping
        selectors : list/tuple of (str, or tuple args)
            Each contains either types, meta column names, column names or regex-compliant strings

        Raises
        ------
        TypeException
            selectors must be of type {list, tuple}
        ValueException
            If terms overlap within selector groups

        Returns
        -------
        self

        See Also
        --------
        cache : Adds a cache element to `selectors_`.
        cache_k : Adds k cache elements to `selectors_`.
        """
        # for each selector, get the group view.
        if isinstance(selectors, (list, tuple)):
            cnames = [self.view(sel) for sel in selectors]
        else:
            raise TypeError("'selectors' must be of type {list, tuple}")

        igrid = intersection_grid(cnames)
        if igrid.shape[0] == 0:
            new_grid = pd.concat([pd.Series(n, index=val) for n, val in zip(selectors, cnames)], sort=False, axis=0)
            new_grid.name = name
        else:
            raise ValueError("shared terms: {} discovered for meta_map.".format(igrid))
        # merge into meta
        self.meta_[name] = object_to_categorical(new_grid, selectors)
        # store meta_map for future reference.
        self.mapper_[name] = selectors
        return self

    @_actionable
    def sort_columns(self, by="alphabet", ascending=True):
        """Sorts `df_` using vast selection criteria.

        Parameters
        -------
        by : str or list/tuple of str, optional
            Sorts columns based on information in `meta_`, or by alphabet, or by index.
            Accepts {'alphabet', 'colnames'} as additional options. 'colnames' is `index`
        ascending : bool or list/tuple of bool, optional
            Sort ascending vs descending.
            If list/tuple, specify multiple ascending/descending combinations.

        Raises
        ------
        ValueException
            If the length of `by` does not equal the length of `ascending`, in list instance.
        TypeException
            If `by` is not of type {str, list, tuple}

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
    def expand(self, column: str, sep: str = ","):
        """Expand out a 'stacked' id column to a longer-form DataFrame.

        Expands out a 'stacked' id column to a longer-form DataFrame, and re-merging
        the data back in.

        Parameters
        ----------
        column : str
            The name of the column to expand, must be of datatype [object]
        sep : str, optional
            The separating string to use.

        Raises
        ------
        ValueError
            If `column` not found in `df_` or `meta_`, or `column` is not stackable

        Returns
        -------
        self

        See Also
        --------
        shrink : Expands out a 'unstacked' id column to a shorter-form DataFrame.
        """
        if column not in self.df_.columns:
            raise ValueError("column '{}' not found in df".format(column))
        if not self.meta_.loc[column, "is_potential_stacker"]:
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
    def shrink(self, column: str, sep: str = ","):
        """Expand out a 'unstacked' id column to a shorter-form DataFrame.

        Shrinks down a 'duplicated' id column to a shorter-form dataframe, and re-merging
        the data back in.

        Parameters
        -------
        column : str
            The name of the duplicated column to shrink, must be of datatype [object]
        sep : str, optional
            The separating string to add.

        Raises
        ------
        ValueError
            If `column` not found in `df_` or `meta_`, or `column` is not shrinkable

        Returns
        -------
        self

        See Also
        --------
        expand : Expands out a 'stacked' id column to a longer-form DataFrame.
        the data back in.
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
    def split_categories(self, column: str, sep: str = ",", renames=None):
        """Split a column into N categorical variables to be associated with df_.

        Parameters
        ----------
        column : str
            The name of the column to split, must be of datatype [object], and contain values sep inside
        sep : str, optional
            The separating string to add.
        renames : None or list of str, optional
            If list of str, must be the same dimension as expanded columns

        Raises
        ------
        ValueError
            `column` not found in `df_` column names

        Returns
        -------
        self
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

    def compute(self, pipe=None, inplace: bool = False, update_meta: bool = False):
        """Execute a pipeline on `df_`.

        Computes a pipeline to the MetaPanda object. If there are no parameters, it computes
        what is stored in the pipe_ attribute, if any.

        .. note:: the `meta_` attribute is **refreshed** after a call to `compute`, if `update_meta`

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        -------
        pipe : str, Pipe, list of 3-tuple, (function name, *args, **kwargs), optional
            A set of instructions expecting function names in MetaPanda and parameters.
            If None, computes the stored pipe_.current pipeline.
            If str, computes the stored pipe_.<name> pipeline.
            If Pipe object, computes the elements in that class.
        inplace : bool, optional
            If True, applies the pipe inplace, else returns a copy. Default has now changed
            to return a copy.
        update_meta : bool, optional
            If True, resets the meta after the pipeline completes.

        Returns
        -------
        self/copy

        See Also
        --------
        compute_k : Executes `k` pipelines on `df_`, in order.
        """
        if pipe is None:
            # use self.pipe_
            pipe = self.pipe_["current"]
            self.pipe_["current"] = []
        if inplace:
            # computes inplace
            self.mode_ = "instant"
            self._apply_pipe(pipe)
            # reset meta here
            if update_meta:
                self._reset_meta()
            return self
        else:
            # full deepcopy, including dicts, lists, hidden etc.
            cpy = self.copy()
            # compute on cop
            cpy.compute(pipe, inplace=True, update_meta=update_meta)
            return cpy

    def compute_k(self, pipes, inplace: bool = False):
        """Execute `k` pipelines on `df_`, in order.

        Computes multiple pipelines to the MetaPanda object, including cached types such as `.current`

        .. note:: the `meta_` attribute is **refreshed** after a call to `compute_k`.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        --------
        pipes : list of (str, list of (list of 3-tuple, (function name, *args, **kwargs))
            A set of instructions expecting function names in MetaPanda and parameters.
            If empty, computes nothing.
        inplace : bool, optional
            If True, applies the pipes inplace, else returns a copy.

        Returns
        -------
        self/copy

        See Also
        --------
        compute : Executes a pipeline on `df_`.
        """
        # join and execute
        return self.compute(join(*pipes), inplace=inplace)

    def write(self, filename: str = None, with_meta: bool = False, *args, **kwargs):
        """Save a MetaPanda to disk.

        .. warning:: Not affected by `mode_` attribute.

        Parameters
        -------
        filename : str, optional
            The name of the file to save, or None it will create a JSON file with Data.
            Accepts filename that end in [.csv, .json]
            If None, writes a JSON file using `name_` attribute.
        with_meta : bool, optional
            If true, saves metafile also, else doesn't
        *args : list, optional
            Arguments to pass to pandas.to_csv, not used with JSON file
        **kwargs : dict, optional
            Keywords to pass to pandas.to_csv, not used with JSON file

        Raises
        ------
        IOException
            File doesn't end in {'json', 'csv'}

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
