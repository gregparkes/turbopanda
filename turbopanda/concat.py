#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 2 2020

@author: gparkes

Handles concatenation of multiple MetaPandas as a replacement to pd.concat
"""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import pandas as pd
import warnings

# locals
from .metapanda import MetaPanda
from .utils import check_list_type, instance_check
from .deprecator import deprecated


__all__ = ('concat')


@deprecated('0.1.8', '0.2.0', 'merge')
def concat(mdfs, name=None, axis=0, **kwargs):
    """
    Concatenates multiple MetaPandas along a given axis. In addition updates to
    preserve meta-information wherever possible. This only applies when axis=1.

    .. TODO: deprecated:: 0.1.8
        This function will be removed in 0.2.0, use turb.merge instead.

    Parameters
    ----------
    mdfs : list of MetaPanda/DataFrame
        The sets of MetaPandas to join together. Accepts pandas.DataFrames also
        but no metadata is used in this instance.
    name : str, optional
        The name of the concatenated dataset, uses the name of the first element otherwise
    axis : int or str, optional
        'index': 0, 'columns': 1
    kwargs : dict, optional
        Additional keyword arguments to pass down to pd.concat

    Returns
    -------
    mdf : pd.MetaPanda
        The concatenated DataFrame.
    """
    instance_check(mdfs, (list, tuple))
    check_list_type(mdfs, MetaPanda)
    nombre = name if name is not None else mdfs[0].name_

    if axis in [0, 'index']:
        data = pd.concat([mdf.df_ for mdf in mdfs], axis=axis, **kwargs)
        # wrap in MetaPanda and return
        mdf = MetaPanda(data, name=nombre)
        # copy over meta
        mdf._meta = mdfs[0].meta_
        return mdf
    elif axis in [1, 'columns']:
        return NotImplemented
    else:
        raise ValueError("axis must be in [0, 1, 'axis', 'columns']")
