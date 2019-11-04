#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:41:59 2019

@author: gparkes

A selection of miscallenaous functions.
"""
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from IPython.display import Image

__all__ = ["cache", "draw"]

def cache(filename="example1.csv"):
    """
    Provides automatic .csv caching for pandas.DataFrames.

    use as:
    @cache("file.csv")
    def f(x):
        return pd.Series(x)

    f([1, 2, 3])
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be of type [str]")
    if not filename.endswith(".csv"):
        raise ValueError("filename '{}' must end with .csv extension".format(filename))

    # define decorator
    def decorator(func):
        # parameters passed to this function.
        def caching_function(*args, **kwargs):
            if os.path.isfile(filename):
                return pd.read_csv(filename, index_col=0)
            else:
                df = func(*args, **kwargs)
                if isinstance(df, pd.DataFrame):
                    df.to_csv(filename)
                    return df
                else:
                    warnings.warn("returned object from func not of type [pd.DataFrame], not saved.", FutureWarning)
                    return df
        return caching_function
    return decorator


def draw(filename="image1.png"):
    """
    A function for calculating the drawing function (which can be expensive)
    and quickly cache-out an image.

    use as:
    @draw("bar1.png")
    def f(x):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4], 'k')
        return fig

    f([11,23])
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be of type [str]")
    if not filename.endswith(".png"):
        raise ValueError("filename '{}' must end with .png extension".format(filename))

    def decorator(func):
        def caching_function(*args, **kwargs):
            if os.path.isfile(filename):
                return Image(filename)
            else:
                fig = func(*args, **kwargs)
                if isinstance(fig, plt.Figure):
                    fig.savefig(filename)
                    return
                else:
                    warnings.warn("returned object from func not of type [plt.Figure], not saved.", FutureWarning)
                    return
        return caching_function
    return decorator


def chunk(filename, func, chunks=50000, usecolumns=None, *args):
    """
    Chunk a large .csv file into stages, using a function
    to 'filter' each subgroup.

    Parameters
    -------
    filename : str
        name of .csv file
    func : function
        The function to call to filter
    chunks : int
        Size of chunks to use (number of rows)
    usecolumns : list
        List of columns to keep, or None
    *args : list
        additional args to pass to func

    returns
    -------
    x : pd.DataFrame/Series
        concatenated results of the chunked-file
    """
    # must be at least chunks length
    if usecolumns is not None:
        chunkset = pd.read_csv(filename, index_col=0, iterator=True,
                               chunksize=chunks, usecols=usecolumns)
    else:
        chunkset = pd.read_csv(filename, index_col=0, iterator=True,
                               chunksize=chunks)
    # calculate filter, and concatenate
    return pd.concat([func(i_chunk, *args) for i_chunk in chunkset])
