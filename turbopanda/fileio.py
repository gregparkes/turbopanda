#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

__all__ = ["read", "read_mp", 'read_raw_json']

import glob
import json
import itertools as it

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

    Note that if multiple files are selected, they are returned in ALPHABETICAL ORDER, not
    necessarily the order in the file directory. If a list of names is passed, this is
    sorted so as to match the filename ordering returned.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: [csv, CSV, xls, xlsx, XLSX, json, hdf]
        .json is a special use case and will use the MetaPanda format, NOT the pd.read_json function.
        filename now accepts glob-compliant input to read in multiple files if selected.
    name : str, list, optional
        A custom name to use for the MetaPanda, else the filename is used. Where this is a list, this
        is sorted to alphabetically match the filename.
    args : list
        Additional args to pass to pd.read_[ext]/MetaPanda()
    kwargs : dict
        Additional args to pass to pd.read_[ext]MetaPanda()

    Returns
    ------
    mdf : list, MetaPanda
        A MetaPanda object. Returns a list of objects if filename is glob-like and
        selects multiple files.
    """
    instance_check(filename, str)

    # use the glob package to allow for unix-like searching. Sorted alphabetically
    glob_name = sorted(glob.glob(filename))
    if len(glob_name) == 0:
        raise IOError("No files selected with filename {}".format(filename))
    else:
        # maps the filetype to a potential pandas function.
        pandas_types = ["csv", "CSV", "xls", "xlsx", "XLSX", "hdf"]

        def ext(s):
            return s.rsplit(".", 1)[-1]

        def fetch_db(fl, n=None):
            if ext(fl) in pandas_types:
                return MetaPanda.from_pandas(fl, n, *args, **kwargs)
            elif ext(fl) == 'json':
                return MetaPanda.from_json(fl, name=n, **kwargs)
            else:
                raise IOError("file ending '{}' not recognized, must end with {}".format(fl, pandas_types + ['json']))

        if isinstance(name, (list, tuple)):
            ds = [fetch_db(f, n) for f, n in it.zip_longest(glob_name, sorted(name))]
        elif isinstance(name, str):
            ds = [fetch_db(f, name) for f in glob_name]
        else:
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


def read_raw_json(filename):
    """
    Reads in a raw JSON file.

    Parameters
    ----------
    filename : str
        A relative/absolute link to the file, with .json optionally provided.

    Returns
    -------
    d : dict
        The JSON object found in the file.
    """
    instance_check(filename, str)

    with open(filename, "r") as f:
        mp = json.load(f)
        f.close()
    return mp
