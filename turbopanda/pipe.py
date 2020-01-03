#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 2 2020

@author: gparkes

An object for handling pipelines in conjunction with MetaPanda
"""
from sklearn import preprocessing
from pandas import to_numeric

from .utils import integer_to_boolean, object_to_categorical, is_n_value_column


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
        return int(s)
    elif _attempt_float_cast(s):
        return float(s)
    elif _is_bool_true(s):
        return True
    elif _is_bool_false(s):
        return False
    else:
        return s


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


class Pipe(object):
    """
    A basic list-like object that allows users to create, manipulate
    and execute functions to a given object/class.
    """

    def __init__(self, *args):
        """
        Creates a 'pipeline' for you using relative shorthand. Each argument
        is a step in the Pipeline.

        e.g Pipe(["apply_columns", "lower"]) returns simply:
            ('apply_columns', ('lower',), {})

        Pipe also accepts keyword arguments in the form of strings. 'axis=0'

        Parameters
        -------
        args : list of arguments
            A series of arguments which can be converted into a suitable and cheap pipeline

        Returns
        -------
        self
        """
        self._p = []
        for arg in args:
            self._p.append(_single_pipe(arg))

    @classmethod
    def raw(cls, p):
        """
        Defines a Pipe using straight raw input.

        Defined as:
            [<function name>, (<function args>), {<function keyword args}], ...

        Parameters
        ----------
        p : list
            List of raw arguments as defined above.

        Returns
        -------
        pipe : Pipe
            fresh Pipe object
        """
        obj = cls()
        obj._p = p
        return obj

    """ ############ PROPERTIES ################### """

    @property
    def p(self):
        return self._p

    """ ############ OVERLOADED FUNCTIONS ############## """

    def __repr__(self):
        return "Pipe(n_elements={})".format(len(self.p))

    def __str__(self):
        return str(self.p)

    """ ############ PUBLIC ACCESSIBLE PIPELINES TO PLUG-AND-PLAY .... ############### """

    @classmethod
    def ml_regression(cls, mp, x_s, y_s, preprocessor="scale"):
        """
        Creates a 'delay' pipe that will make your data 'machine-learning-ready'.
        Prepares for a regression-based model.

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
        pipe : Pipe
            The pipeline to perform on mp
        """
        if hasattr(preprocessing, preprocessor):
            # try to get function
            preproc_f = getattr(preprocessing, preprocessor.lower())
        else:
            raise ValueError("preprocessor function '{}' not found in sklearn.preprocessing".format(preprocessor))

        # out of the x-features, we only preprocess.scale continuous features.
        return cls.raw([
            # drop objects, ids columns
            ("drop", (object, ".*id$", ".*ID$", "^ID.*", "^id.*"), {}),
            # drop any columns with single-value-type in
            ("apply", ("drop", mp.view(is_n_value_column),), {"axis": 1}),
            # drop missing values in y
            ("apply", ("dropna",), {"axis": 0, "subset": mp.view(y_s)}),
            # fill missing values with the mean
            ("transform", (lambda x: x.fillna(x.mean()), x_s), {}),
            # apply standard scaling to X
            ("transform", (preproc_f,), {"selector": x_s, "whole": True}),
        ])

    @classmethod
    def clean(cls):
        """
        Pipe that cleans the pandas.DataFrame. Applies a number of transformations which (attempt to) reduce the datatype,
        cleaning column names.

        Parameters
        --------

        Returns
        -------
        pipe : Pipe
            The pipeline object
        """
        return cls.raw([
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
        ])

    @classmethod
    def no_id(cls):
        """
        Pipe that drops ID-like columns from the dataset.
        Includes regex string search for terms like id, ID, etc.

        Parameters
        --------

        Returns
        -------
        pipe : list
            The pipeline object
        """
        return cls.raw([
            # drops elements of type object, id selectors
            ("drop", (object, ".*id$", ".*ID$", "^ID.*", "^id.*"), {})
        ])
