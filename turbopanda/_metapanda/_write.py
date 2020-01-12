#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Responsible for writing MetaPanda files to disk."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import json
from typing import Optional


def _write_csv(self, filename: str, with_meta: bool = False, *args, **kwargs):
    self.df_.to_csv(filename, sep=",", *args, **kwargs)
    if with_meta:
        directory, jname, ext = split_file_directory(filename)
        self.meta_.to_csv(directory + "/" + jname + "__meta.csv", sep=",")


def _write_json(self, filename: str):
    # columns founded by meta_map are dropped
    redundant_meta = union(meta_columns_default(), list(self.mapper_.keys()))
    reduced_meta = self.meta_.drop(redundant_meta, axis=1)
    # saving_dict

    compile_string = '{"data":%s,"meta":%s,"name":%s,"cache":%s,"mapper":%s,"pipe":%s}' % (
        self.df_.to_json(path_or_buf=None, double_precision=12),
        reduced_meta.to_json(path_or_buf=None, double_precision=12) if reduced_meta.shape[1] > 0 else "",
        self.name_,
        str(self.selectors_),
        str(self.mapper_),
        str(self.pipe_)
    )
    # determine file name.
    fn = filename if filename is not None else self.name_ + '.json'
    with open(fn, "w") as f:
        json.dump(compile_string, f)


def _write_hdf(self, filename: str):
    """TODO: Saves a file in special HDF5 format."""
    return NotImplemented


def write(self,
          filename: Optional[str] = None,
          with_meta: bool = False,
          *args,
          **kwargs) -> "MetaPanda":
    """Save a MetaPanda to disk.

    .. warning:: Not affected by `mode_` attribute.

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
        # default: make a json file
        self._write_json(self.name_ + ".json")
    elif filename.endswith(".csv"):
        self._write_csv(filename, with_meta, *args, **kwargs)
    elif filename.endswith(".json"):
        self._write_json(filename)
    else:
        raise IOError("Doesn't recognize filename or type: '{}', must end in [csv, json]".format(filename))
    return self
