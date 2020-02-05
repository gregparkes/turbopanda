#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# locals
from typing import Optional, Dict, Union, List

from ._metapanda import MetaPanda
from .utils import instance_check
from ._deprecator import deprecated


__all__ = ("read", 'read_raw_json')


def read(filename: str,
         name: Optional[Union[str, List[str]]] = None,
         *args,
         **kwargs) -> Union[MetaPanda, List[MetaPanda]]:
    """Reads in a data source from file and creates a MetaPanda object from it.

    Note that if multiple files are selected, they are returned in ALPHABETICAL ORDER, not
    necessarily the order in the file directory. If a list of `name` is passed, this is
    sorted so as to match the filename ordering returned.

    Parameters
    ----------
    filename : str
        A relative/absolute link to the file, with extension provided.
        Accepted extensions: {'csv'', 'xls', 'xlsx', 'sql', 'json', 'hdf'}
        .json is a special use case and will use the MetaPanda format, NOT the pd.read_json function.
        .hdf is a special use and stores both df_ and meta_ attributes.
        `filename` now accepts glob-compliant input to read in multiple files if selected.
    name : str/list of str, optional
        A custom name to use for the MetaPanda, else `filename` is used. Where this is a `list`, this
        is sorted to alphabetically match `filename`.
    args : list, optional
        Additional args to pass to pd.read_[ext]/MetaPanda()
    kwargs : dict, optional
        Additional args to pass to pd.read_[ext]MetaPanda()

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
    # imports for this function
    import glob
    import itertools as it

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

        def ext(s):
            """Extracts the file extension (in lower)"""
            return s.rsplit(".", 1)[-1].lower()

        def fetch_db(fl: str, n=None) -> "MetaPanda":
            """Fetches the appropriate datafile set."""
            if ext(fl) in pandas_types:
                return MetaPanda.from_pandas(fl, n, *args, **kwargs)
            elif ext(fl) == 'json':
                return MetaPanda.from_json(fl, name=n, **kwargs)
            else:
                raise ValueError(
                    "non-pandas file ending '{}' not recognized, must end with {}".format(fl, pandas_types))

        if isinstance(name, (list, tuple)):
            ds = [fetch_db(f, n) for f, n in it.zip_longest(glob_name, sorted(name))]
        elif isinstance(name, str):
            ds = [fetch_db(f, name) for f in glob_name]
        else:
            ds = [fetch_db(f) for f in glob_name]
        # if we have more than one element, return the list, else just return ds
        return ds if len(ds) > 1 else ds[0]


@deprecated("0.1.9", "0.2.2", "read")
def read_raw_json(filename: str) -> Dict:
    """
    Reads in a raw JSON file.

    .. deprecated:: 0.1.9
        `read_raw_json` will be removed in 0.2.2, use `read` instead.

    TODO: deprecate 'read_raw_json' in version 0.2.2

    Parameters
    ----------
    filename : str
        A relative/absolute link to the file, with .json optionally provided.

    Returns
    -------
    d : dict
        The JSON object found in the file.
    """
    instance_check(filename, str)

    import json

    with open(filename, "r") as f:
        mp = json.load(f)
        f.close()
    return mp
