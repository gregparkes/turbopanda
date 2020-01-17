#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for reading in files."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import turbopanda as turb


def _toy_dataset_1():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": ['Ha', 'Ho', 'He'],
        "c": [True, False, True],
        "d": np.random.rand(3),
    })


def test_init():
    # initialize from dataset
    df = _toy_dataset_1()
    # valid input
    mdf = turb.MetaPanda(df)
    # attribute checks
    assert hasattr(mdf, "name_")
    assert hasattr(mdf, "df_")
    assert hasattr(mdf, "meta_")
    assert hasattr(mdf, "n_")
    assert hasattr(mdf, "p_")
    assert hasattr(mdf, "selectors_")
    assert hasattr(mdf, "pipe_")
    assert hasattr(mdf, "columns")
    assert hasattr(mdf, "mapper_")
    assert hasattr(mdf, "mode_")
    # type checks
    assert isinstance(mdf, turb.MetaPanda)
    assert isinstance(mdf.name_, str)
    assert isinstance(mdf.df_, pd.DataFrame)
    assert isinstance(mdf.meta_, pd.DataFrame)
    assert isinstance(mdf.n_, int)
    assert isinstance(mdf.p_, int)
    assert isinstance(mdf.columns, pd.Index)
    assert isinstance(mdf.selectors_, dict)
    assert isinstance(mdf.mapper_, dict)
    assert isinstance(mdf.mode_, str)
    # check size
    assert mdf.n_ == 3
    assert mdf.p_ == 4


def test_dtypes():
    # valid input
    mdf = turb.MetaPanda(_toy_dataset_1())
    # correct types [uint8, object, bool, float64]
    # has downcasted to np.uint8
    assert mdf.df_['a'].dtype == np.uint8
    assert mdf.df_['b'].dtype in (object, pd.CategoricalDtype)
    assert mdf.df_['c'].dtype == np.bool
    assert mdf.df_['d'].dtype == np.float64
    # check dtypes function
    assert hasattr(mdf, "dtypes")
    dt = mdf.dtypes()
    assert isinstance(dt, pd.Series)
    assert len(dt) == 4
    dt2 = mdf.dtypes(grouped=False)
    # more to come...


if __name__ == '__main__':
    x = _toy_dataset_1()
    y = turb.MetaPanda(x)
