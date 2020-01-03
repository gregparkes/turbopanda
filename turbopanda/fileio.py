#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

__all__ = ["read", "read_mp"]

import glob
from .metapanda import MetaPanda
from .utils import instance_check


def read(filename, name=None, *args, **kwargs):
    """
    Reads in a datafile and creates a MetaPanda object from it.

        NEW IN 0.1.7 - Glob support
            filename now supports the glob library,
            which uses unix-like pathname pattern expression.
            This allows you to read in multiple files simultaenously.
                e.g "*.json" reads in every JSON file in the local directory.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: [csv, CSV, xls, xlsx, XLSX, json, hdf]
        .json is a special use case and will use the MetaPanda format, NOT the pd.read_json function.
        filename now accepts glob-compliant input to read in multiple files if selected.
    name : str, optional
        A custom name to use for the MetaPanda, else the filename is used
    args : list
        Additional args to pass to pd.read_[ext]
    kwargs : dict
        Additional args to pass to pd.read_[ext]

    Returns
    ------
    mdf : list, MetaPanda
        A MetaPanda object. Returns a list of objects if filename is glob-like and
        selects multiple files.
    """
    instance_check(filename, str)
    # use the glob package to allow for unix-like searching.
    glob_name = glob.glob(filename)
    if len(glob_name) == 0:
        raise IOError("No files selected with filename {}".format(filename))
    else:
        # maps the filetype to a potential pandas function.
        pandas_types = ["csv", "CSV", "xls", "xlsx", "XLSX", "hdf"]

        def ext(s):
            return s.rsplit(".", 1)[-1]

        def fetch_db(fl):
            if ext(fl) in pandas_types:
                return MetaPanda.from_pandas(fl, name, *args, **kwargs)
            elif ext(fl) == 'json':
                return MetaPanda.from_json(fl)
            else:
                raise IOError("file ending '{}' not recognized, must end with {}".format(fl, pandas_types + ['json']))

        # iterate and return if present
        ds = [fetch_db(f) for f in glob_name]
        # if we have more than one element, return the list, else just return ds
        return ds if len(ds) > 1 else ds[0]


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
