#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
from copy import deepcopy
from typing import Union, List

from pandas import to_numeric, Index
from sklearn import preprocessing

# locals
from .utils import boolean_to_integer, object_to_categorical, \
    is_n_value_column, instance_check
from .custypes import PipeTypeRawElem, ListTup, PipeTypeCleanElem


def _is_float_cast(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _type_cast_argument(s: str):
    """Casts a string 'type' into it's proper basic type.

    Works on integers, floats and bool.
    """
    if s.isdigit():
        return int(s)
    elif _is_float_cast(s):
        return float(s)
    elif s in {'true', 'True'}:
        return True
    elif s in {'false', 'False'}:
        return False
    else:
        return s


def is_pipe_structure(pipe: ListTup[PipeTypeRawElem]) -> bool:
    """Check whether the pipe passed fits our raw pipe structure."""
    for p in pipe:
        if len(p) != 3:
            raise ValueError("pipe of length 3 is of length {}".format(len(pipe)))
        # element 1: string
        instance_check(pipe[0], str)
        instance_check(pipe[1], (list, tuple))
        instance_check(pipe[2], dict)
    return True


def _single_pipe(argument: PipeTypeCleanElem) -> List:
    """Converts a single command line into pipeable code for MetaPanda."""
    instance_check(argument[0], str)

    pipe_command = [argument[0]]
    pipe_d = {}
    pipe_t = []
    for arg in argument[1:]:
        if isinstance(arg, str):
            # if it's a string, check whether it's a param with = **kwarg
            if arg.find("=") != -1:
                key, value = arg.split("=", 1)
                value = _type_cast_argument(value)
                # attempt to convert sp[1] from str to int, float, bool or other basic type.
                pipe_d[key] = value
            else:
                # otherwise tupleize it as an *arg
                pipe_t.append(arg)
        elif isinstance(arg, dict):
            # set the keywords within the solo dictionary
            pipe_d = arg
        else:
            pipe_t.append(arg)

    # assemble and return
    pipe_command.append(tuple(pipe_t))
    pipe_command.append(pipe_d)
    return pipe_command


class Pipe(object):
    """An object for handling pipelines of data.

    A basic list-like object that allows users to create, manipulate
    and execute functions to a given object/class.

    Attributes
    ----------
    p_ : tuple of PipeTypeElems
        The steps of each argument in the pipeline. See `custypes.py` for accepted PipeTypeElems.

    Methods
    -------
    copy : None
        Deepcopy this current object into a new one.
    None
    """

    def __init__(self, *args: PipeTypeCleanElem):
        """Define a Pipeline for your object.

        Creates a pipeline for you using relative shorthand. Each argument
        is a step in the `Pipe`.

        Pipe can accept a bunch of arguments corresponding to pipe steps:
        >>> import turbopanda as turb
        >>> turb.Pipe(('apply_columns', 'lower'))
        ('apply_columns', ('lower',), {})

        The leading argument for each step must be a string
        naming the function you wish to call, with following arguments as *arg.
        If a dictionary is passed, this becomes the **kwarg parameter.
        Pipe also accepts keyword arguments in the form of strings. 'axis=0'

        Parameters
        -------
        args : str, Pipe, list of 3-tuple, (function name, *args, **kwargs)
            A set of instructions expecting function names in MetaPanda and parameters.

        Returns
        -------
        self
        """
        _p = []
        for arg in args:
            _p.append(_single_pipe(arg))
        # convert to tuple and store
        self._p = tuple(_p)

    @classmethod
    def raw(cls, p: ListTup[PipeTypeRawElem]) -> "Pipe":
        """Defines a Pipe using straight raw input.

        Defined as:
            [<function name>, (<function args>), {<function keyword args}], ...

        Parameters
        ----------
        p : list/tuple
            List of raw arguments as defined above.

        Returns
        -------
        pipe : Pipe
            fresh Pipe object
        """
        # perform check
        if is_pipe_structure(p):
            obj = cls()
            obj._p = p
            return obj
        else:
            raise ValueError("pipe: {} not of correct structure".format(p))

    """ ##################### PROPERTIES ################################ """

    @property
    def p(self) -> ListTup[PipeTypeRawElem]:
        """Return the raw pipeline."""
        return self._p

    """ ############ OVERLOADED FUNCTIONS ############################### """

    def __repr__(self) -> str:
        """Represent the object as a string."""
        return "Pipe(n_elements={})".format(len(self.p))

    """ ############### PUBLIC METHODS ################################### """

    def copy(self):
        """Copy this object into a new object."""
        return deepcopy(self)

    """ ############ PUBLIC ACCESSIBLE PIPELINES TO PLUG-AND-PLAY .... ############### """

    @classmethod
    def ml_regression(cls,
                      mp,
                      x_s: Union[ListTup[str], Index],
                      y_s: Union[str, ListTup[str], Index],
                      preprocessor: str = "scale") -> "Pipe":
        """The default pipeline for Machine Learning Regression problems.

        Parameters
        --------
        mp : MetaPanda
            The MetaPanda object
        x_s : list of str, pd.Index
            A list of x-features
        y_s : str/list of str, pd.Index
            A list of y-feature(s)
        preprocessor : str, optional
            Name of preprocessing: default 'scale', choose from
                {'power_transform', 'minmax_scale', 'maxabs_scale', 'robust_scale',
                 'quantile_transform', 'scale', 'normalize'}

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
        return cls.raw((
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
        ))

    @classmethod
    def clean(cls, with_drop: bool = True) -> "Pipe":
        """Pipeline to clean a pandas.DataFrame.

        Pipe that cleans the pandas.DataFrame. Applies a number of transformations which (attempt to) reduce the datatype,
        cleaning column names. Transformations include:

        * Dropping columns with only one unique value type.
        * Converting columns to numeric where possible
        * Converting object-columns to categoricals where possible
        * Converting boolean columns to `uint8` type
        * Stripping column names of spaces
        * Renaming column names to eliminate tabs, whitespace and `-`

        Parameters
        --------
        TODO: with_drop : bool, optional
            If True, drops columns containing only one data value.

        Returns
        -------
        pipe : Pipe
            The pipeline object
        """

        return cls.raw((
            # drop columns with only one value in
            ("drop", (lambda x: x.dropna().unique().shape[0] == 1,), {}),
            # shrink down data types where possible.
            ("apply", ("transform", to_numeric,), {"errors": "ignore", "downcast": "unsigned"}),
            # convert int to categories
            ("transform", (lambda x: object_to_categorical(x),), {"selector": object}),
            # convert booleans to uint8
            ("apply", ('transform', boolean_to_integer), {}),
            # strip column names
            ("apply_columns", ("strip",), {}),
            # do some string stripping
            ("rename", ([(" ", "_"), ("\t", "_"), ("-", "")],), {}),
        ))

    @classmethod
    def no_id(cls) -> "Pipe":
        """Pipeline to drop ID-like columns.

        Drops columns containing {'ID', 'id', object}.

        Returns
        -------
        pipe : list
            The pipeline object
        """
        return cls(('drop', object, ".*id$", ".*ID$", "^ID.*", "^id.*"))
