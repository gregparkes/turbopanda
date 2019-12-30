#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:29:46 2019

@author: gparkes

Caching functions that take a function which returns a MetaPanda object
and caches it to file, ready to re-install if re-run.
"""
import os
import warnings
from pandas import DataFrame

from .metapanda import MetaPanda
from .fileio import read
from .utils import instance_check, belongs

__all__ = ["cache"]


def cache(filename="example1.json"):
    """
    Provides automatic [.json|.csv] caching for turb.MetaPanda or pandas.DataFrames

    example:
        @cache_metapanda('metafile.json')
        def f(x):
            return turb.MetaPanda(x)

    Custom function MUST return a pandas.DataFrame OR turb.MetaPanda object.

    Parameters
    --------
    filename : str
        The name of the file to cache to, or read from. This is fixed. Accepts only .csv and
        .json files currently.

    Returns
    -------
    mp : turb.MetaPanda
        The MetaPanda object
    """
    # check it is string
    instance_check(filename, str)
    # check that file ends with json or csv
    belongs(filename.rsplit(".")[-1], ["json", "csv"])

    # define decorator
    def decorator(func):
        def caching_function(*args, **kwargs):
            # if we find the file
            if os.path.isfile(filename):
                # read it in
                return read(filename)
            else:
                # returns MetaPanda or pandas.DataFrame
                mpf = func(*args, **kwargs)
                if isinstance(mpf, MetaPanda):
                    # save file
                    mpf.write(filename)
                elif isinstance(mpf, DataFrame):
                    # save
                    mpf.to_csv(filename)
                else:
                    warnings.warn("returned object from cache not of type [pd.DataFrame, turb.MetaPanda], not cached",
                                  FutureWarning)
                return mpf

        return caching_function

    return decorator
