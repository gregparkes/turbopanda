#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:17:26 2019

@author: gparkes
"""
import numpy as np
from pandas import to_numeric
from sklearn import preprocessing

from .utils import integer_to_boolean, object_to_categorical, is_n_value_column

__all__ = ["ml_regression_pipe", "clean_pipe"]


def ml_regression_pipe(mp, x_s, y_s, preprocessor="scale"):
    """
    Creates a 'delay' pipe that will make your data 'machine-learning-ready'.

    Parameters
    --------
    mp : MetaPanda
        The MetaPanda object
    x_s : list of str, pd.Index
        A list of x-features
    y_s : str/list of str, pd.Index
        A list of y-feature(s)
    preprocessor : str
        Name of preprocessing: default 'scale', choose from
            [power_transform, minmax_scale, maxabs_scale, robust_scale,
             quantile_transform, scale, normalize]
            Choose from sklearn.preprocessing.

    Returns
    -------
    pipe : list
        The pipeline to perform on mp
    """
    if hasattr(preprocessing, preprocessor):
        # try to get function
        preproc_f = getattr(preprocessing, preprocessor.lower())
    else:
        raise ValueError("preprocessor function '{}' not found in sklearn.preprocessing".format(preprocessor))

    # out of the x-features, we only preprocess.scale continuous features.
    return [
        # drop objects, ids columns
        ("drop", (object, "_id$", "_ID$", "^counter"), {}),
        # drop any columns with single-value-type in
        ("apply", ("drop", mp.view(is_n_value_column),), {"axis": 1}),
        # drop missing values in y
        ("apply", ("dropna",), {"axis": 0, "subset": mp.view(y_s)}),
        # fill missing values with the mean
        ("transform", (lambda x: x.fillna(x.mean()), x_s), {}),
        # apply standard scaling to X
        ("transform", (preproc_f,), {"selector": x_s, "whole": True}),
    ]


def clean_pipe():
    """
    Pipe that cleans the pandas.DataFrame. Applies a number of transformations which (attempt to) reduce the datatype,
    cleaning column names.

    Parameters
    --------
    None

    Returns
    -------
    pipe : list
        The pipeline to perform on mp
    """
    return [
        # shrink down data types where possible.
        ("apply", ("transform", to_numeric,), {"errors": "ignore", "downcast": "unsigned"}),
        # convert 2-ints into boolean columns
        ("transform", (lambda x: integer_to_boolean(x),), {"selector": int}),
        # convert int to categories
        ("transform", (lambda x: object_to_categorical(x),), {"selector": object}),
        # strip column names
        ("apply_columns", ("strip",), {}),
        # strip string object columns
        ("transform", (lambda x: x.str.strip(),), {"selector": object}),
        # do some string stripping
        ("rename", ([(" ", "_"), ("\t", "_"), ("-", "")],), {}),
    ]