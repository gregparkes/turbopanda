#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the MetaPanda object."""

import os
import hashlib
import json
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# from turbopanda._pipe import Pipe
from turbopanda._deprecator import deprecated, deprecated_param
from turbopanda.utils._files import split_file_directory
from turbopanda.utils._sets import join
from ._drop_values import drop_columns
from ._inspect import inspect
from ._types import SelectorType


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
    shape: tuple (int, int)
        The number of rows and columns in data
    memory_ : str
        String-formatted representation of memory consumption in megabytes
    options_ : tuple
        A list of options for selectors
    selectors_ : dict
        Maps unique name (key) to cached selected groups of columns (value)
    mapper_ : dict
        Maps unique name (key) to key of cached selected column groups (value)

    Methods
    -------
    head(k=5)
        A wrapper for pandas.DataFrame.head()
    dtypes(grouped=True)
        Determines the data type of each column in `df_`
    printf()
        Prints the raw data in full pandas format
    view(selector)
        Views a selection of columns in `df_`
    search(selector)
        View the intersection of search terms, for columns in `df_`.
    view_not(selector)
        Views the non-selected columns in `df_`.
    select(sel_str)
        Views the selection of columns using an eval-like string
    patsy(X, y=None)
        Describes a statistical model in basic patsy language
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
    downcast()
        Casts all variables in higher form to lower form if possible
    keep(selector)
        Keeps the selected columns from `df_` only
    cache(name, selector)
        Adds a cache element to `selectors_`
    cache_k(selectors)
        Adds k cache elements to `selectors_`
    rename_axis(ops, selector=None, axis=1)
        Performs a chain of .str.replace operations on `df_.columns`
    add_prefix(pref, selector=None)
        Adds a string prefix to selected columns
    add_suffix(suf, selector=None)
        Adds a string suffix to selected columns
    transform(func, selector=None,  method='transform', whole=False, args, kwargs)
        Performs an inplace transformation to a group of columns within `df_`.
    transform_k(ops)
        Performs multiple inplace transformations to a group of columns within `df_`
    aggregate(func, name=None, selector=None, keep=False)
        Perform an inplace column-wise aggregation with a selector.
    aggregate_k(func, names=None, selectors=None, keep=False)
        Perform inplace column-wise aggregations to multiple selectors.
    meta_map(name, selectors)
        Maps a group of selectors with an identifier, in `mapper_`
    update_meta(None)
        Forces an update to reset the `meta_` attribute.
    meta_split_category(cat)
        Splits category into k boolean columns in `meta_` to use for selection.
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
                 dataset: pd.DataFrame,
                 name: Optional[str] = None,
                 with_clean: bool = True,
                 with_warnings: bool = False):
        """Define a MetaPanda frame.

        Creates a Meta DataFrame with the raw data and parametrization of
        the DataFrame by its grouped columns.

        Parameters
        ----------
        dataset : pd.DataFrame, optional
            The raw data set to create as a MetaDataFrame. If None, assumes the dataset will be set later.
        name : str, optional
            Gives the MetaDataFrame a name, which comes into play with merging, for instance
        with_clean : bool, optional
            If True, uses Pipe.clean() to perform minor preprocessing on `dataset`
        with_warnings : bool, optional
            If True, prints warnings when strange things happen, for instance during selection
        """
        self._with_warnings = with_warnings
        self._with_clean = with_clean

        # selectors saved
        self._select = {}
        # pipeline arguments
        self._pipe = {"current": []}
        # columns added to meta_map
        self._mapper = {}
        # set empty meta
        self._meta = None
        self._source = ""
        # set using property
        self.df_ = dataset
        self.name_ = name if name is not None else 'DataSet'

    """ ########################### IMPORTED FUNCTIONS ####################################### """

    # inspecting columns
    from ._inspect import view_not, view, select
    # dropping rows
    from ._drop_values import drop, keep
    # shadowed columns
    from ._shadow import head, dtypes, copy, info
    # saving files
    from ._write import write, _write_csv, _write_json, _write_pickle, printf
    # application to pandas.api functions
    from ._apply import apply, apply_index, apply_columns, _apply_function, \
        _apply_index_function, _apply_column_function
    # caching selectors and pipes
    from ._caching import cache, cache_pipe, cache_k
    # computing pipelines
    from ._compute import compute_k, compute, _apply_pipe
    # reshaping operations with strings
    from ._string_reshape import expand, shrink, split_categories
    # patsy-like functionality
    from ._patsy import patsy
    # renaming columns
    from ._name_axis import rename_axis, add_suffix, add_prefix
    # metadata operations
    from ._metadata import sort_columns, meta_split_category, meta_map, update_meta
    # transformation operations
    from ._transform import transform, transform_k, aggregate, aggregate_k, downcast

    """ ############################ STATIC FUNCTIONS ######################################## """

    @classmethod
    def from_pandas(cls, filename: str, name: str = None, *args, **kwargs):
        """Read in a MetaPanda from a comma-separated version (csv) file.

        Parameters
        -------
        filename : str
            A relative/absolute link to the file, with extension provided.
            Accepted extensions: {'csv', 'xls', 'xlsx', 'XLSX'', 'sql'}
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
            "csv": pd.read_csv, "xls": pd.read_excel,
            "xlsx": pd.read_excel, "sql": pd.read_sql
        }

        # split filename into parts and check
        directory, jname, ext = split_file_directory(filename)

        df = file_ext_map[ext](filename, *args, **kwargs)
        # create MetaPanda
        mp = cls(df, name=name) if name is not None else cls(df, name=jname)
        # set the source
        mp.source_ = filename
        # return
        return mp

    @classmethod
    def from_json(cls, filename: str, **kwargs):
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
        with open(filename, "rb") as f:
            # reads in JSON string from the file
            mp = f.read()
        # convert into Python objects - dict
        obj = json.loads(mp)
        # extract checksum, compare to current object.
        if "checksum" not in obj.keys():
            raise IOError("'checksum' not found in JSON file: {}".format(filename))
        # go over attributes and assign where available
        if "data" in obj.keys():
            df = pd.DataFrame.from_dict(obj["data"])
            df.index.name = "counter"
            df.columns.name = "colnames"
            # assign to self
            mpf = cls(df, **kwargs)
        else:
            raise ValueError("column 'data' not found in MetaPandaJSON")

        if "cache" in obj.keys():
            mpf._select = obj["cache"]
        if "mapper" in obj.keys():
            mpf._mapper = obj["mapper"]
        if "name" in obj.keys() and "name" not in kwargs.keys():
            mpf.name_ = obj["name"]
        if "pipe" in obj.keys():
            mpf._pipe = obj["pipe"]

        # checksum check.
        chk = hashlib.sha256(json.dumps(mpf.df_.columns.tolist()).encode()).hexdigest()
        if obj['checksum'] != chk:
            raise IOError("checksum stored: %s doesn't match %s" % (obj['checksum'], chk))
        # apply additional metadata columns
        mpf.update_meta()
        # set the source
        mpf.source_ = filename
        # return
        return mpf

    """ ############################## OVERRIDDEN OPERATIONS ###################################### """

    def __copy__(self):
        warnings.warn("the copy constructor in 'MetaPanda' has no functionality.", RuntimeWarning)

    def __getitem__(self, *selector: SelectorType):
        """Fetch a subset determined by the selector."""
        # we take the columns that are NOT this selection, then drop to keep order.
        sel = inspect(self.df_, self.meta_, self.selectors_, selector, join_t="union", mode='view')
        if sel.size > 0:
            # drop anti-selection to maintain order/sorting
            return self.df_[sel].squeeze()

    def __setitem__(self, cname, series):
        """Assigns a new column to the MetaPanda."""
        self.df_[cname] = series
        # update the meta
        self.update_meta()

    def __delitem__(self, *selector: SelectorType):
        """Delete columns determined by the selector."""
        # drops columns inplace
        drop_columns(self.df_, self.meta_, self.view(selector))

    def __repr__(self) -> str:
        """Represent the object in a stringed format."""
        p = self.df_.shape[1] if self.df_.ndim > 1 else 1
        """
        OPTIONS are:
        - S: contains selectors
        - M: contains mapper(s)
        """
        opt_s = "S" if len(self.selectors_) > 0 else ''
        opt_m = "M" if len(self.mapper_) > 0 else ''
        opts = "[" + opt_s + opt_m + ']'
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
    def source_(self) -> str:
        """Returns the source file associated with this dataset."""
        return self._source

    @source_.setter
    def source_(self, source):
        if isinstance(source, str):
            self._source = source
            # if the file doesn't exist, write this object to the named source
            if not os.path.isfile(source):
                # write to the source, assuming file extension at the end
                self.write(source)
        else:
            raise TypeError("source_ must be of type str, not {}".format(type(source)))

    @property
    def df_(self) -> pd.DataFrame:
        """Fetch the raw dataset."""
        return self._df

    @df_.setter
    def df_(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            # apply the 'cleaning pipeline' to this.
            self._df = df
            # define meta
            self.update_meta()
            if "colnames" not in self._df.columns:
                self._df.columns.name = "colnames"
            if "counter" not in self._df.columns:
                self._df.index.name = "counter"
        else:
            raise TypeError("'df' must be of type [pd.DataFrame]")

    @property
    def meta_(self) -> pd.DataFrame:
        """Fetch the dataframe with meta information."""
        return self._meta

    @property
    @deprecated_param("0.2.8", "n_", "0.3", "renamed to `n`")
    def n_(self) -> int:
        """Fetch the number of rows/samples within the df_ attribute."""
        return self.df_.shape[0]

    @property
    @deprecated_param("0.2.8", "p_", "0.3", "renamed to `p`")
    def p_(self) -> int:
        """Fetch the number of dimensions within the df_ attribute."""
        return self.df_.shape[1]

    @property
    def shape(self) -> Tuple:
        """Fetches the shape of the dataset."""
        return self.df_.shape

    @property
    def columns(self) -> pd.Index:
        """Forward on 'columns' property."""
        return self.df_.columns

    """ Additional meta information """

    @property
    def selectors_(self) -> Dict[str, Any]:
        """Fetch the cached selectors available. This also includes boolean columns found in `meta_`."""
        return self._select

    @property
    def options_(self) -> Tuple:
        """Fetch the available selector options cached in this object."""
        return tuple(join(self._select, self.meta_.columns[self.meta_.dtypes == np.bool]))

    @property
    def memory_(self) -> str:
        """Fetch the memory consumption of the MetaPanda."""
        df_m = self.df_.memory_usage(deep=False).sum() / 1000000
        meta_m = self.df_.memory_usage(deep=False).sum() / 1000000
        return "{:0.3f}MB".format(df_m + meta_m)

    @property
    def name_(self) -> str:
        """Fetch the name of the MetaPanda."""
        return self._name

    @name_.setter
    def name_(self, n: str):
        if isinstance(n, str):
            # if name is found as a column name, block it.
            if n in self.df_.columns:
                raise ValueError("name: {} for df_ found as a column attribute; not allowed!".format(n))
            self._name = n
        else:
            raise TypeError("'name_' must be of type str")

    @property
    @deprecated_param("0.2.8", "pipe_", "0.3", "pipe object is being phased out")
    def pipe_(self) -> Dict[str, Any]:
        """Fetch the cached pipelines."""
        return self._pipe

    @property
    def mapper_(self) -> Dict[str, Any]:
        """Fetch the mapping between unique name and selector groups."""
        return self._mapper
