#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

import os
import pandas as pd

__all__ = ["read", "write"]

from .metapanda import MetaPanda


def read(filename, name=None, metafile=None, *args, **kwargs):
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
    args : list
        Additional args to pass to pd.read_[ext]
    kwargs : dict
        Additional args to pass to pd.read_[ext]

    Returns
    ------
    mdf : MetaPanda
        A MetaPanda object.
    """
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
        mp = MetaPanda(df, name=name)
    else:
        mp = MetaPanda(df, name=jname)
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


def write(mp, *args, **kwargs):
    """
    Writes a MetaPanda, including meta-data to disk.
    """
    # uses the name stored in mp
    mp.write(*args, **kwargs)
    return
