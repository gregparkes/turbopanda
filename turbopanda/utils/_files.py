#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods utility file-related functions."""

from typing import Any, List


__all__ = ("list_dir", "split_file_directory", 'insert_prefix', 'insert_suffix')


def list_dir(obj: Any) -> List:
    """Lists all public functions, classes within a list directory."""
    return [a for a in dir(obj) if not a.startswith("__") and not a.startswith("_")]


def split_file_directory(filename: str):
    """Breaks down the filename pathway into constitute parts.

    Parameters
    --------
    filename : str
        The filename full string

    Returns
    -------
    directory : str
        The directory linking to the file
    jname : str
        The name of the file (without extension)
    ext : str
        Extension type
    """
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
    return directory, jname, ext


def insert_prefix(filename: str, ins: str) -> str:
    """Inserts a string to the start of a file name.

    Parameters
    ----------
    filename : str
        The filename full string
    ins : str
        The insertable string

    Returns
    -------
    new_filename : str
        The new filename full string
    """
    _f = split_file_directory(filename)
    return "/".join([_f[0], ins + _f[1]]) + "." + _f[-1]


def insert_suffix(filename: str, ins: str) -> str:
    """Inserts a string to the end of a file name.

    Parameters
    ----------
    filename : str
        The filename full string
    ins : str
        The insertable string

    Returns
    -------
    new_filename : str
        The new filename full string
    """
    _f = split_file_directory(filename)
    return "/".join([_f[0], _f[1] + ins]) + "." + _f[-1]
