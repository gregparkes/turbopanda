#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:48:53 2019

@author: gparkes
"""

import pandas as pd

from .metapanda import MetaPanda

__all__ = ["read"]


def read(filename, *args, **kwargs):
    """
    Reads in a datafile and creates a MetaPanda object from it.

    Parameters
    -------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: [csv, xls, xlsx, html, json, hdf, sql]
    args : list
        Additional args to pass to pd.read_[ext]
    kwargs : dict
        Additional args to pass to pd.read_[ext]

    Returns
    ------
    mdf : MetaPanda
        A metapanda object.
    """
    file_ext_map = {
        "csv": pd.read_csv, "xls": pd.read_excel, "xlsx": pd.read_excel,
        "html": pd.read_html, "json": pd.read_json, "hdf": pd.read_hdf,
        "sql": pd.read_sql
    }

    ext = filename.split(".")[-1]
    df = file_ext_map[ext](filename, *args, **kwargs)
    # map to MetaPanda
    return MetaPanda(df)
