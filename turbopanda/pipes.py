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


__all__ = ["pipe", "ml_regression_pipe", "clean_pipe"]


def _attempt_float_cast(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_bool_true(s):
    return s in ["true", "True"]


def _is_bool_false(s):
    return s in ["false", "False"]


def _type_cast_argument(s):
    if s.isdigit():
        value = int(s)
    elif _attempt_float_cast(s):
        value = float(s)
    elif _is_bool_true(s):
        value = True
    elif _is_bool_false(s):
        value = False
    return value


def _single_pipe(argument):
    """
    Converts a single command line into pipeable code for MetaPanda.
    """
    pipe_command = [argument[0]]
    pipe_d = {}
    pipe_t = []
    for arg in argument[1:]:
        if isinstance(arg, str):
            # if it's a string, check whether it's a param with = **kwarg
            if arg.find("=") != -1:
                key, value = arg.split("=",1)
                value = _type_cast_argument(value)
                # attempt to convert sp[1] from str to int, float, bool or other basic type.
                pipe_d[key] = value
            else:
                # otherwise tupleize it as an *arg
                pipe_t.append(arg)
        else:
            pipe_t.append(arg)
    pipe_command.append(tuple(pipe_t))
    pipe_command.append(pipe_d)
    return tuple(pipe_command)


################################## PUBLIC FUNCTIONS ####################################################


def pipe(arguments):
    """
    Creates a 'pipeline' for you using relative shorthand.

    e.g pipe("apply_columns", "lower") returns simply:
        ('apply_columns', ('lower',), {})

    Parameters
    -------
    arguments : list of arguments
        A series of arguments which can be converted into a suitable and cheap pipeline

    Returns
    -------
    pipe : list of arguments
        A full pipe that can be passed to MetaPanda.compute
    """
    mpipe = []
    for arg in arguments:
        mpipe.append(_single_pipe(arg))
    return mpipe


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
        # drop columns with only one value in
        ("drop", (lambda x: x.dropna().unique().shape[0] == 1,), {}),
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