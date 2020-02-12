#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to writing out files in MetaPanda."""

import pandas as pd
import hashlib
import json
from typing import Optional

from ._metadata import default_columns
from turbopanda.utils import split_file_directory, union


__all__ = ('write', '_write_csv', '_write_json', '_write_hdf')


def _write_csv(self, filename: str, with_meta: bool = False, *args, **kwargs):
    self.df_.to_csv(filename, sep=",", *args, **kwargs)
    if with_meta:
        directory, jname, ext = split_file_directory(filename)
        self.meta_.to_csv(directory + "/" + jname + "__meta.csv", sep=",")


def _write_json(self, filename: str):
    # update meta information
    self.update_meta()
    # columns founded by meta_map are dropped
    redundant_meta = union(list(default_columns().keys()), list(self.mapper_.keys()))
    reduced_meta = self.meta_.drop(redundant_meta, axis=1, errors='ignore')
    # encode data
    stringed_data = self.df_.to_json(double_precision=12)
    stringed_meta = reduced_meta.to_json(double_precision=12) if reduced_meta.shape[1] > 0 else "{}"
    # generate checksum - using just the column names.
    checksum = hashlib.sha256(json.dumps(self.df_.columns.tolist()).encode()).hexdigest()
    # compilation string
    compile_string = '{"data":%s,"meta":%s,"name":%s,"cache":%s,"mapper":%s,"pipe":%s,"checksum":%s}' % (
        stringed_data,
        stringed_meta,
        json.dumps(self.name_),
        json.dumps(self.selectors_),
        json.dumps(self.mapper_),
        json.dumps(self.pipe_),
        json.dumps(checksum),
    )
    # determine file name.
    fn = filename if filename is not None else self.name_ + '.json'
    with open(fn, "wb") as f:
        f.write(compile_string.encode())


def _write_hdf(self, filename: str):
    """Saves a file in special HDF5 format.

    TODO: Implement `_write_hdf` function.
    """
    return NotImplemented


def write(self,
          filename: Optional[str] = None,
          with_meta: bool = False,
          *args,
          **kwargs) -> "MetaPanda":
    """Save a MetaPanda to disk.

    Parameters
    -------
    filename : str, optional
        The name of the file to save, or None it will create a JSON file with Data.
        Accepts filename that end in [.csv, .json]
        If None, writes a JSON file using `name_` attribute.
    with_meta : bool, optional
        If true, saves metafile also, else doesn't
    *args : list, optional
        Arguments to pass to pandas.to_csv, not used with JSON file
    **kwargs : dict, optional
        Keywords to pass to pandas.to_csv, not used with JSON file

    Raises
    ------
    IOException
        File doesn't end in {'json', 'csv'}

    Returns
    -------
    self
    """
    if filename is None:
        self._write_json(self.name_ + ".json")
    elif filename.endswith(".csv"):
        self._write_csv(filename, with_meta, *args, **kwargs)
    elif filename.endswith(".json"):
        self._write_json(filename)
    else:
        raise IOError("Doesn't recognize filename or type: '{}', must end in [csv, json]".format(filename))
    return self