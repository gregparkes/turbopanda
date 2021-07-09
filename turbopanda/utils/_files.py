#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods utility file-related functions."""

import glob
import os
import warnings
from typing import Any, List, Iterable

from ._sets import join

__all__ = (
    "list_dir",
    "split_file_directory",
    "insert_prefix",
    "insert_suffix",
    "get_file_expanded",
    "check_file_path"
)


def _is_filepath_object(f):
    """Returns whether f is a filepath type object."""
    try:
        glob.glob(f)
        return True
    except TypeError:
        return False


def get_file_expanded(files: List[str]) -> Iterable:
    """Given a list of filenames, get the associated found files."""
    return join(*[glob.glob(f) for f in files if _is_filepath_object(f)])


def list_dir(obj: Any) -> List:
    """Lists all public functions, classes within a list directory."""
    return [a for a in dir(obj) if not a.startswith("__") and not a.startswith("_")]


def list_dir_object(obj: Any) -> List[str]:
    """Lists all of the non-hidden directories"""
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
    fs = filename.replace("\\","/").rsplit("/", 1)
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
    lhs, rhs = filename.rsplit(".", 1)
    return lhs + ins + "." + rhs


def check_file_path(path: str,
                    create_folder: bool = False,
                    raise_error: bool = True,
                    verbose: int = 0) -> bool:
    """Checks along a filepath and raises errors if one of the files doesn't exist.

    Parameters
    ----------
    path : str
        The filepath to the desired file.
    create_folder : bool
        Creates a folder with warning if not found at each step
    raise_error : bool
        Raises an error if a folder isn't found, does not create folder
    verbose : int
        Prints statements at each folder step

    Returns
    -------
    result : bool
        True or False
    """

    # handle case where create_folder and raise_error are true
    if raise_error and create_folder:
        warnings.warn("`raise_error` and `create_folder` cannot both be true, defaulting to `raise_error`")

    def _remove_garb(x):
        return x != "" and x != ".." and x.find(".") == -1

    # get a list of folders in order
    folders = list(filter(_remove_garb, path.split("/")))

    if verbose > 1:
        print("folders: " + folders)

    # iterate through each folder subtype and check
    for f in folders:
        constructed_string = path[:path.find(f) + len(f)]
        if verbose > 1:
            print("folder step: " + constructed_string)
        # does this subpart exist
        if not os.path.isdir(constructed_string):
            absp = os.path.abspath(constructed_string)
            # now create, raise or warn
            if raise_error:
                # raise an error
                raise FileNotFoundError("directory at '{}' does not exist".format(absp))
            elif create_folder:
                if verbose > 0:
                    print("creating folder at '{}'".format(absp))
                os.mkdir(constructed_string)
            else:
                warnings.warn("directory at '{}' does not exist".format(absp), UserWarning)
    return True
