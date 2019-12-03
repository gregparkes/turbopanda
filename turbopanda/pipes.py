#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:17:26 2019

@author: gparkes
"""

from sklearn import preprocessing

from .metadata import is_single_value_column

__all__ = ["ml_regression_pipe", "ml_continuous_pipe"]


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
    """
    # fetch columns
    y_f = mp.view(y_s)
    # columns with only 1 value in
    one_col = mp.view(is_single_value_column)
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
        ("apply", ("drop", one_col), {"axis": 1})
        # drop missing values in y
        ("apply", ("dropna",), {"axis": 0, "subset": y_f}),
        # fill missing values with the mean
        ("transform", (lambda x: x.fillna(x.mean()), x_s), {}),
        # apply standard scaling to X
        ("transform", (preproc_f,), {"selector": x_s, "whole": True}),
    ]


def ml_continuous_pipe(mp, x_s, y_s, preprocessor="scale"):
    """
    Creates a 'delay' pipe that will make your data 'machine-learning-ready'.
    Only keeps continuous columns, i.e of type float.

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
    """
    # fetch columns
    y_f = mp.view(y_s)
    # columns with only 1 value in
    one_col = mp.view(is_single_value_column)

    if hasattr(preprocessing, preprocessor):
        # try to get function
        preproc_f = getattr(preprocessing, preprocessor.lower())
    else:
        raise ValueError("preprocessor function '{}' not found in sklearn.preprocessing".format(preprocessor))

    return [
        # drop objects, ids columns
        ("drop", (object, "_id$", "_ID$", "^counter$"), {}),
        # drop any columns with single-value-type in
        ("apply", ("drop", one_col), {"axis": 1})
        # drop missing values in y
        ("apply", ("dropna",), {"axis": 0, "subset": y_f}),
        # fill missing values with the mean
        ("transform", (lambda x: x.fillna(x.mean()), x_s), {}),
        # apply standard scaling to X
        ("transform", (preproc_f,), {"selector": x_s, "whole": True}),
    ]