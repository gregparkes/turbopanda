#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

import os
import json
import pandas as pd

__all__ = ["read", "read_mp"]

from .metapanda import MetaPanda


def read(filename, name=None, metafile=None, key=None, *args, **kwargs):
    """
    Reads in a datafile and creates a MetaPanda object from it.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: [csv, xls, xlsx, html, json, hdf, sql]
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
    if filename.endswith(".csv"):
        return MetaPanda.from_csv(filename, name, metafile, key, *args, **kwargs)
    elif filename.endswith(".json"):
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
