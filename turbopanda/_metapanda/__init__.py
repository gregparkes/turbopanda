#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This is the base file for MetaPanda object."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# imports
import sys
import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any
import pandas as pd

sys.path.append("../")
from .custypes import SelectorType
from .pipe import Pipe


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
                 dataset: pd.DataFrame,
                 name: Optional[str] = None,
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

    """ IMPORT METHODS """

    # overloaded direct operations
    from ._overloaded import head, dtypes, copy
    # selection types
    from ._selection import view, view_not, search, _selector_group, _get_selector, _get_selector_item
    # cache
    from ._cache import cache, cache_k, cache_pipe
    # renaming columns
    from ._rename_columns import _rename_axis, rename, add_prefix, add_suffix
    # dropping/keeping columns
    from ._dropper import keep, drop
    # application functions to pandas
    from ._apply import apply, apply_columns, apply_index
    # some interesting reshape functions
    from ._reshape import shrink, expand, split_categories, sort_columns
    # metadata creation, analysis
    from ._metadata import _reset_meta, update_meta, meta_map
    # transform functions
    from ._transform import transform, transform_k, filter_rows
    # computing pipelines
    from ._compute import compute, compute_k
    # writing files to disk using write()
    from ._write import write, _write_csv, _write_json, _write_hdf

    """ STATIC METHODS GO HERE """

    @classmethod
    def from_pandas(cls, filename: str, name: str = None, *args, **kwargs) -> "MetaPanda":
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
    def from_json(cls, filename: str, **kwargs) -> "MetaPanda":
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
        mpf.update_meta()

        return mpf

    @classmethod
    def from_hdf(cls, filename: str, **kwargs) -> "MetaPanda":
        """TODO: Read in a MetaPanda from a custom HDF5 file.

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

    """ HIDDEN OVERLOADED FUNCTIONS """

    def __copy__(self):
        warnings.warn("the copy constructor in 'MetaPanda' has no functionality.", RuntimeWarning)

    def __getitem__(self,
                    selector: Union[SelectorType, Tuple[SelectorType, ...], List[SelectorType]]):
        """Fetch a subset determined by the selector."""
        # we take the columns that are NOT this selection, then drop to keep order.
        sel = _selector_group(self, selector)
        if sel.size > 0:
            # drop anti-selection to maintain order/sorting
            return self.df_[sel].squeeze()

    def __delitem__(self, *selector: SelectorType):
        """Delete columns determined by the selector."""
        # drops columns inplace
        self.drop(selector)

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

    """ PROPERTIES """

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
            _reset_meta(self)
            # compute cleaning.
            if self._with_clean:
                self.compute(Pipe.clean(), inplace=True)

            if "colnames" not in self._df.columns:
                self._df.columns.name = "colnames"
            if "counter" not in self._df.columns:
                self._df.index.name = "counter"
        elif isinstance(df, (pd.Series, pd.DataFrameGroupBy)):
            # again, we'll just pretend the user knows what they're doing...
            self._df = df
        else:
            raise TypeError("'df' must be of type [pd.Series, pd.DataFrame, DataFrameGroupBy]")

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

    """ Additional meta information """

    @property
    def selectors_(self) -> Dict[str, Any]:
        """Fetch the cached selectors available."""
        return self._select

    @property
    def memory_(self) -> str:
        """Fetch the memory consumption of the MetaPanda."""
        return "{:0.3f}MB".format(calc_mem(self.df_) + calc_mem(self.meta_))

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
    def mode_(self) -> str:
        """Choose from {'instant', 'delay'}."""
        return self._mode

    @mode_.setter
    def mode_(self, mode: str):
        if mode in ("instant", "delay"):
            self._mode = mode
        else:
            raise ValueError("'mode' must be ['instant', 'delay'], not '{}'".format(mode))

    @property
    def pipe_(self) -> Dict[str, Any]:
        """Fetch the cached pipelines."""
        return self._pipe

    @property
    def mapper_(self) -> Dict[str, Any]:
        """Fetch the mapping between unique name and selector groups."""
        return self._mapper
