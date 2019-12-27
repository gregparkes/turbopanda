#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

import os

__all__ = ["read", "read_mp"]

from .metapanda import MetaPanda
from .utils import instance_check


def read(filename, name=None, metafile=None, key=None, *args, **kwargs):
    """
    Reads in a datafile and creates a MetaPanda object from it.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: [csv, CSV, xls, xlsx, XLSX, json, hdf]
        .json is a special use case and will use the MetaPanda format, NOT the pd.read_json function.
    name : str, optional
        A custom name to use for the MetaPanda, else the filename is used
    metafile : str, optional
        An associated meta file to join into the MetaPanda, else if None,
        attempts to find the file, otherwise just creates the raw default.
    key : None, str, optional
        Sets one of the columns as the 'primary key' in the Dataset
    args : list
        Additional args to pass to pd.read_[ext]
    kwargs : dict
        Additional args to pass to pd.read_[ext]

    Returns
    ------
    mdf : MetaPanda
        A MetaPanda object.
    """
    instance_check(filename, str)
    if not os.path.isfile(filename):
        raise IOError("file at '{}' does not exist".format(filename))
    # maps the filetype to a potential pandas function.
    pandas_types = [".csv", ".CSV", ".xls", ".xlsx", ".XLSX", ".hdf"]
    # iterate and return if present
    for ft in pandas_types:
        if filename.endswith(ft):
            return MetaPanda.from_pandas(filename, name, metafile, key, *args, **kwargs)
    if filename.endswith(".json"):
        return MetaPanda.from_json(filename)
    else:
        raise IOError("file ending '{}' not recognized, must end with [csv, json]".format(filename))


def read_mp(filename):
    """
    Reads in a MetaPanda object from it. Note that
    this method only works if you read in a JSON file with
    the format generated from a matching
    write_mp() method.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with .json optionally provided.

    Returns
    ------
    mdf : MetaPanda
        A MetaPanda object.
    """
    return MetaPanda.from_json(filename)
