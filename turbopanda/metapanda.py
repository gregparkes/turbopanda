#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:55:38 2019

@author: gparkes
"""

import re
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.stats import kstest

from .utils import *


class MetaPanda(object):
    """
    A pandas.DataFrame, but with a few extra goodies. Contains meta-functionality
    for the pandas.DataFrame, including complex handling of grouped-columns.
    """

    def __init__(self,
                 dataset,
                 cat_thresh=20,
                 default_remove_single_col=True):
        """
        Creates a Meta DataFrame with the raw data and parameterization of
        the dataframe by its grouped columns.

        Parameters
        -------
        dataset : pd.DataFrame
            The raw dataset to create as a metadataframe.
        cat_thresh : int
            The threshold until which 'category' variables are not created
        default_remove_one_column : bool
            Decides whether to drop columns with a single unique value in (default True)
        """

        self._cat_thresh = cat_thresh
        self._def_remove_single_col = default_remove_single_col
        # set using property
        self.df_ = dataset
        self._select = {}


    ############################### HIDDEN FUNCTIONS ##################################


    def __make_boolean__(self, c, rename_dict):
        self.df_[c] = self.df_[c].astype(np.bool)
        rename_dict[c] = "is_" + c

    def __make_categorical__(self, uniques, c):
        c_cat = CategoricalDtype(np.sort(uniques), ordered=True)
        self.df_[c] = self.df_[c].astype(c_cat)

    def __column_not_float__(self, c):
        return self.df_[c].dtype not in [np.float64, np.float32, np.float, np.float16, float]

    def __column_float__(self, c):
        return self.df_[c].dtype in [np.float64, np.float32, np.float, np.float16, float]

    def __column_is_object__(self, c):
        return self.df_[c].dtype in [object, pd.CategoricalDtype]

    def __unique_ID_column__(self, c):
        # a definite algorithm for determining a unique column ID
        return self.df_[c].is_unique if self.__column_not_float__(c) else False

    def __potential_ID_column__(self, c, thresh=0.5):
        return self.df_[c].unique().shape[0] / self.df_.shape[0] > thresh if self.__column_not_float__(c) else False

    def __potential_stacker_column__(self, c, regex=";|\t|,", thresh=0.5):
        return self.df_[c].str.contains(regex).sum() > thresh if self.__column_is_object__(c) else False

    def __normality__(self, c):
        return kstest(self.df_[c].dropna().values, "norm")[1] < 0.05 if self.__column_float__(c) else False


    def _categorize(self):
        col_renames = {}
        # iterate over all column names.
        for c in self.df_.columns:
            if self.df_[c].dtype in [np.int64, object]:
                un = self.df_[c].unique()
                if un.shape[0] == 1:
                    if self._def_remove_single_col:
                        self.df_.drop(c, axis=1, inplace=True)
                    else:
                        # convert to bool
                        self.__make_boolean__(c, col_renames)
                    # there is only one value in this column, remove it.
                elif un.shape[0] == 2:
                    if np.all(np.isin([0, 1], un)):
                        # use boolean
                        self.__make_boolean__(c, col_renames)
                    else:
                        # turn into 2-factor categorical if string, etc.
                        self._make_categorical_(un, c)
                elif un.shape[0] <= self._cat_thresh:
                    # convert to categorical if string, int, etc.
                    self.__make_categorical__(un, c)

        # apply all global rename changes to the column.
        self.df_.rename(columns=col_renames, inplace=True)


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


    def _construct_metadata(self):
        # step 1. construct a dataframe based on the column names as an index.
        colnames = self.df_.columns
        # step 2. find potential unique ID columns.
        is_uniq = [self.__unique_ID_column__(c) for c in self.df_]
        # step 3. find potential ID columns.
        is_id = [self.__potential_ID_column__(c) for c in self.df_]
        # step 4. find potential stacked ID columns - stackers include [;,\t]
        is_stacked = [self.__potential_stacker_column__(c) for c in self.df_]
         # step 5. normality of floating-based columns
        is_normal = [self.__normality__(c) for c in self.df_]

        # FINAL - return all as a meta_ option.
        self._meta = pd.DataFrame({
            "is_unique": is_uniq, "potential_id": is_id, "potential_stacker":is_stacked,
            "is_norm":is_normal
        }, index=colnames)


    def _get_selector_item(self, selector, raise_error=True):
        if selector in [
                float, int, bool, "category", "float", "int", "bool", "boolean",
                np.float64, np.int64, object
            ]:
            # if it's a string option, convert to type
            string_to_type = {"category": pd.CategoricalDtype, "float": np.float64,
                              "int": np.int64, "bool": np.bool, "boolean": np.bool,
                              float: np.float64, int: np.int64, object:object,
                              np.float64:np.float64, np.int64:np.int64, bool:np.bool}
            nk = string_to_type[selector]
            return self.df_.columns[self.df_.dtypes.eq(nk)]
        # check if selector is a callable object (i.e function)
        elif callable(selector):
            # call the selector, assuming it takes a pandas.DataFrame argument. Must
            # return a boolean Series.
            ser = selector(self.df_)
            # perform check
            is_boolean_series(ser)
            # check lengths
            not_same = self.df_.columns.symmetric_difference(ser.index)
            # if this exists, append these true cols on
            if not_same.shape[0] > 0:
                ns = pd.concat([pd.Series(True, index=not_same), ser], axis=0)
                return self.df_.columns[ns]
            else:
                return self.df_.columns[ser]
        elif isinstance(selector, str):
            # check if the key is in the meta_ column names
            if selector in self.meta_:
                return self.df_.columns[self.meta_[selector]]
            elif selector in self._select:
                # recursively go down the stack, and fetch the string selectors from that.
                return self._get_selector(self._select[selector], raise_error)
            # check if key does not exists in df.columns
            elif selector not in self.df_:
                # try regex
                col_fetch = [c for c in self.df_.columns if re.search(selector, c)]
                if len(col_fetch) > 0:
                    return pd.Index(col_fetch, dtype=object,
                                    name=self.df_.columns.name, tupleize_cols=False)
                elif raise_error:
                    raise ValueError("selector '{}' yielded no matches.".format(selector))
                else:
                    return pd.Index([], name=self.df_.columns.name)
            else:
                # we assume it's in the index, and we return it, else allow pandas to raise the error.
                return selector
        else:
            raise TypeError("selector type '{}' not recognized".format(type(selector)))

    def _get_selector(self, selector, raise_error=True, select_join="OR"):
        if isinstance(selector, (tuple, list)):
            # iterate over all selector elements and get pd.Index es.
            for s in selector:
                s_groups = [self._get_selector_item(s,raise_error) for s in selector]
                if select_join == "AND":
                    return chain_intersection(*s_groups)
                elif select_join == "OR":
                    return chain_union(*s_groups)
            # by default, use intersection for AND, union for OR
        else:
            # just one item, return asis
            return self._get_selector_item(selector,raise_error)


    ############################ OVERRIDEN OPERATIONS #######################################


    def __getitem__(self, selector):
        # firstly check if 'key' is a datatype to keep!#
        return self.df_[self._get_selector(selector)].squeeze()


    def __delitem__(self, selector):
        # drops columns inplace
        selection = self._get_selector(selector)
        if selection.shape[0] > 0:
            self.df_.drop(selection, axis=1, inplace=True)
            # drop meta
            self.meta_.drop(selection, axis=0, inplace=True)


    def __repr__(self):
        return "MetaPanda(DataSet(n={}, p={}, mem={}))".format(
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
            self._levels = self.df_.columns.nlevels if isinstance(self.df_.columns, pd.MultiIndex) else 1
            # if multi-column, concatenate to a single column.
            self._remove_MultiIndex()
            # perform categorization
            self._categorize()
            # strip column names of spaces either side
            self.df_.columns = self.df_.columns.str.strip()
            # strip spaces within text-based features
            self._remove_spaces_in_obj_columns()
            # remove spaces/tabs within the column name.
            self.df_.columns = self.df_.columns.str.replace(" ", "_").str.replace("\t","_").str.replace("-","")
            # sort index
            self.df_.sort_index(inplace=True)
            # construct metadata on columns
            self._construct_metadata()
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
        return self._get_selector(selector, select_join="OR")


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
        selected_list = self._get_selector(selector, select_join="OR")
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
        selection = self._get_selector(selector, raise_error=False, select_join="OR")
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
            raise ValueError("cache name '{}' already exists in .cache, use another".format(name))
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
            curr_cols = sel_cols = self._get_selector(selector)
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
