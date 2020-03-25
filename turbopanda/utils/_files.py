#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods utility file-related functions."""

import glob
from typing import Any, List

from ._error_raise import instance_check
from ._sets import join

__all__ = ("list_dir", "split_file_directory", 'insert_prefix', 'insert_suffix', 'get_file_expanded')


def _is_filepath_object(f):
    """Returns whether f is a filepath type object."""
    try:
        glob.glob(f)
        return True
    except TypeError:
        return False


def get_file_expanded(files: List[str]) -> List:
    """Given a list of filenames, get the associated found files."""
    instance_check(files, (list, tuple))
    return join(*[glob.glob(f) for f in files if _is_filepath_object(f)])


def get_super_directory(dname):
    """Given the directory name, """


def list_dir(obj: Any) -> List:
    """Lists all public functions, classes within a list directory."""
    return [a for a in dir(obj) if not a.startswith("__") and not a.startswith("_")]


def list_dir_object(obj: Any) -> List:
    import string
    """Lists all public objects within a directory."""
    return [a for a in dir(obj) if a[0] in string.ascii_uppercase]


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
