#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

import os
import json
import pandas as pd

__all__ = ["read", "read_mp", "write", "write_mp"]

from .metapanda import MetaPanda
from .utils import instance_check


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
    instance_check(filename, str)
    if not os.path.isfile(filename):
        raise IOError("file at '{}' does not exist".format(filename))

    file_ext_map = {
        "csv": pd.read_csv, "xls": pd.read_excel, "xlsx": pd.read_excel,
        "html": pd.read_html, "json": pd.read_json, "hdf": pd.read_hdf,
        "sql": pd.read_sql, "XLSX": pd.read_excel
    }

    fs = filename.rsplit("/", 1)
    if len(fs) == 0:
        raise ValueError("filename '{}' not recognized".format(filename))
    elif len(fs) == 1:
        directory = "."
        fname = fs[0]
    else:
        directory, fname = fs
    # just the name without the extension
    jname, ext = fname.split(".", 1)

    df = file_ext_map[ext](filename, *args, **kwargs)
    # map to MetaPanda
    if name is not None:
        mp = MetaPanda(df, name=name, key=key)
    else:
        mp = MetaPanda(df, name=jname, key=key)
        name = "_"

    if metafile is not None:
        met = pd.read_csv(metafile, index_col=0, header=0, sep=",")
        mp.meta_ = met
    else:
        # try to find a metafile in the same directory.
        dir_files = os.listdir(directory)
        # potential combination of acceptable names to find
        combs = [jname + "__meta.csv", name + "__meta.csv"]

        for potential_name in combs:
            if potential_name in dir_files:
                met = pd.read_csv(directory + "/" + potential_name, index_col=0, header=0, sep=",")
                # add to mp
                mp.meta_ = met
                return mp

    return mp


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
    instance_check(filename, str)
    # look for attributes 'data', 'meta', 'name', 'pipe' and 'cache'
    if not os.path.isfile(filename):
        raise IOError("file at '{}' does not exist".format(filename))
    # check if ends with .json
    if not filename.endswith(".json"):
        filename += ".json"
    # read in JSON
    with open(filename,"r") as f:
        recvr = json.load(f)
    # check that recvr has all attributes.
    attrs = ["data", "meta", "name", "cache", "pipe"]



def write(mp, *args, **kwargs):
    """
    Writes a MetaPanda, including meta-data to disk.
    """
    # uses the name stored in mp
    mp.write(*args, **kwargs)
    return


def write_mp(mp, filename=None):
    """
    Writes a MetaPanda, including meta-data to JSON file.
    """
    # uses the name stored in mp
    mp.write_json(filename=filename)
    return
