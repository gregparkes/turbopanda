#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the MetaPanda object."""

import json
import pandas as pd
import hashlib
import warnings
from typing import Any, Dict, Union, Tuple, Optional, List
from turbopanda.utils import split_file_directory
from pandas.core.groupby.generic import DataFrameGroupBy

from turbopanda._pipe import Pipe, is_pipe_structure, PipeMetaPandaType
from ._inspect import inspect
from ._types import SelectorType
from ._drop_values import drop_columns


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
           Perform a chain of .str.replace operations on a given `df_` or `meta_` column.
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
       aggregate(func, selectors, keep=False)
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
       eval()
           Evaluates an operation(s) using an expr
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
        # set using property
        self.df_ = dataset
        self.name_ = name if name is not None else 'DataSet'

    """ ########################### IMPORTED FUNCTIONS ####################################### """

    # inspecting columns
    from ._inspect import view_not, view, search
    # dropping rows
    from ._drop_values import drop, keep, filter_rows
    # shadowed columns
    from ._shadow import head, dtypes, copy
    # saving files
    from ._write import write, _write_csv, _write_hdf, _write_json
    # application to pandas.api functions
    from ._apply import apply, apply_index, apply_columns, _apply_function, \
        _apply_index_function, _apply_column_function
    # caching selectors and pipes
    from ._caching import cache, cache_pipe, cache_k
    # computing pipelines
    from ._compute import compute_k, compute, _apply_pipe
    # reshaping operations with strings
    from ._string_reshape import expand, shrink, split_categories
    # renaming columns
    from ._name_axis import rename_axis, rename, add_suffix, add_prefix
    # metadata operations
    from ._metadata import sort_columns, meta_split_category, meta_map, update_meta
    # transformation operations
    from ._transform import transform, transform_k, aggregate, eval

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
        # return
        return mpf

    @classmethod
    def from_hdf(cls, filename: str, **kwargs):
        """Read in a MetaPanda from a custom HDF5 file.

        TODO: Implement `from_hdf` function.

        Reads in a datafile from hdf and creates a MetaPanda object from it.
        There may be issues with storing pipe_, selector_ and
        mapper_ objects due to their complexity.

        Parameters
        -------
        filename : str
            A relative/absolute link to the HDF file.
        **kwargs : dict, optional
            Other keyword arguments to pass to MetaPanda constructor

        Returns
        ------
        mpf : MetaPanda
            A MetaPanda object.
        """
        return NotImplemented

    """ ############################## OVERRIDDEN OPERATIONS ###################################### """

    def __copy__(self):
        warnings.warn("the copy constructor in 'MetaPanda' has no functionality.", RuntimeWarning)

    def __getitem__(self, selector: Union[SelectorType, Tuple[SelectorType, ...], List[SelectorType]]):
        """Fetch a subset determined by the selector."""
        # we take the columns that are NOT this selection, then drop to keep order.
        sel = inspect(self.df_, self.meta_, self.selectors_, selector, join_t='OR')
        if sel.size > 0:
            # drop anti-selection to maintain order/sorting
            return self.df_[sel].squeeze()

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
            # compute cleaning.
            if self._with_clean:
                self.compute(Pipe.clean(with_drop=False), inplace=True)
            if "colnames" not in self._df.columns:
                self._df.columns.name = "colnames"
            if "counter" not in self._df.columns:
                self._df.index.name = "counter"
        elif isinstance(df, DataFrameGroupBy):
            # again, we'll just pretend the user knows what they're doing...
            self._df = df
        else:
            raise TypeError("'df' must be of type [pd.DataFrame, pd.DataFrameGroupBy]")

    @property
    def meta_(self) -> pd.DataFrame:
        """Fetch the dataframe with meta information."""
        return self._meta

    @property
    def n_(self) -> int:
        """Fetch the number of rows/samples within the df_ attribute."""
        return self.df_.shape[0]

    @property
    def p_(self) -> int:
        """Fetch the number of dimensions within the df_ attribute."""
        return self.df_.shape[1]

    @property
    def columns(self) -> pd.Index:
        """Forward on 'columns' property."""
        return self.df_.columns

    """ Additional meta information """

    @property
    def selectors_(self) -> Dict[str, Any]:
        """Fetch the cached selectors available."""
        return self._select

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
    def pipe_(self) -> Dict[str, Any]:
        """Fetch the cached pipelines."""
        return self._pipe

    @property
    def mapper_(self) -> Dict[str, Any]:
        """Fetch the mapping between unique name and selector groups."""
        return self._mapper