#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles basic import functions."""

# future imports
from __future__ import absolute_import, division, print_function

# locals
import glob
from typing import List, Optional, Union

from ._metapanda import MetaPanda
from turbopanda._dependency import is_joblib_installed
from .utils import instance_check, join


def read(
    filename: str, name: Optional[Union[str, List[str]]] = None, **kwargs
) -> Union[MetaPanda, List[MetaPanda]]:
    """Reads in a data source from file and creates a MetaPanda object from it.

    .. note:: multiple files are returned in 'alphabetical order'.



    Parameters
    ----------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: {'csv'', 'xls', 'xlsx', 'sql', 'json', 'pkl'}
        .json is a special use case and will use the MetaPanda format.
        .pkl is a pickable object which will use `joblib` to load in.
    name : str/list of str, optional
        Name to use for the MetaPanda, else `filename` is used.
            If `list`, this is sorted to alphabetically match `filename`.
        Not compatible with .pkl file types.
    kwargs : dict, optional
        Additional args to pass to pd.read_<ext>MetaPanda()

    Raises
    ------
    IOException
        If the no `filename` are selected, or `filename` does not exist
    ValueException
        If `filename` has incorrect file type ending

    Returns
    -------
    mdf : (list of) MetaPanda
        A MetaPanda object. Returns a list of MetaPanda if `filename` is glob-like and
        selects multiple files.
    """
    # checks
    instance_check(filename, str)
    instance_check(name, (type(None), str, list, tuple))

    # use the glob package to allow for unix-like searching. Sorted alphabetically
    glob_name = sorted(glob.glob(filename))
    if len(glob_name) == 0:
        raise IOError("No files selected with filename {}".format(filename))
    else:
        # maps the file type to a potential pandas function.
        pandas_types = ("csv", "xls", "xlsx", "sql")
        extra_types = ("json", "pkl")

        def ext(s):
            """Extracts the file extension (in lower)"""
            return s.rsplit(".", 1)[-1].lower()

        def fetch_db(fl, n=None):
            """Fetches the appropriate datafile set."""
            if ext(fl) in pandas_types:
                return MetaPanda.from_pandas(fl, n, **kwargs)
            elif ext(fl) == "json":
                return MetaPanda.from_json(fl, name=n, **kwargs)
            elif ext(fl) == "pkl":
                # check if joblib is loaded
                if is_joblib_installed(raise_error=True):
                    return joblib.load(fl)
            else:
                raise ValueError(
                    "file ending '.{}' not recognized, must end with {}".format(
                        fl, join(pandas_types, extra_types)
                    )
                )

        if isinstance(name, (list, tuple)):
            ds = list(map(fetch_db, glob_name, sorted(name)))
        elif isinstance(name, str):
            ds = [fetch_db(f, name) for f in glob_name]
        else:
            ds = list(map(fetch_db, glob_name))
        # if we have more than one element, return the list, else just return ds
        return ds if len(ds) > 1 else ds[0]
