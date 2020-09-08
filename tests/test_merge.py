#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for merging samples together."""

# future imports
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

import turbopanda as turb


def toy_dataset_1():
    """Toy dataset 1."""
    return turb.MetaPanda(pd.concat([
        pd.DataFrame(np.random.rand(50, 3), columns=['x1', 'x2', 'x3']),
        pd.DataFrame(np.random.randint(0, 10, size=(50, 2)), columns=['xi1', 'xi2']),
        pd.Series(["LABEL%d" % d for d in range(50)], name='lID')
    ], axis=1, sort=False))


def toy_dataset_2():
    """Toy dataset 2 of size 50."""
    return pd.concat([
        pd.DataFrame(np.random.normal(0, 2, size=(100, 2)), columns=['b1', 'b2']),
        pd.DataFrame(np.random.randint(0, 10, size=(100, 2)), columns=['bj1', 'bj2']),
        pd.Series(['LABEL%d' % d for d in range(100)], name='kID')
    ], sort=False, axis=1)


def toy_dataset_3():
    """Toy dataset 3 of size 14."""
    return turb.MetaPanda(pd.concat([
        pd.DataFrame(np.random.normal(1, 5, size=(14, 2)), columns=['c1', 'c2']),
        pd.Series(['LABEL%d' % d for d in range(14)], name='zID')
    ], sort=False, axis=1))


def toy_dataset_4():
    """Toy dataset 3 of size 14."""
    return pd.Series(np.random.choice(list('abcde'), size=(30,)),
                     index=['LABEL%d' % d for d in range(30)], name='xID').astype('category')


""" Tests begin here """


def test_merge_input_checks():
    df1 = toy_dataset_1()
    df2 = toy_dataset_3()

    # acceptable merge.
    assert turb.merge([df1, df2])
    # one correct, one incorrect
    with pytest.raises(TypeError):
        turb.merge([5, df1])
    # acceptable
    assert turb.merge([df1, df2], name=None)
    with pytest.raises(TypeError):
        turb.merge([df1, df2], name=3.5467)
    with pytest.raises(ValueError):
        turb.merge([df1, df2], how=5)
    with pytest.raises(ValueError):
        # empty input
        turb.merge([])
    # just one input
    with pytest.raises(ValueError):
        turb.merge([df1])
    # test file string inputs
    with pytest.raises(IOError):
        turb.merge("hello boyos")
    with pytest.raises(ValueError):
        turb.merge(['fakefile1.csv', 'fakefile2.csv'])


def test_dataframe_itself():
    df = toy_dataset_2()
    # dataframe + dataframe
    result = turb.merge([df, df])
    # simply a concatenation of itself...
    # returns metapanda
    assert isinstance(result, turb.MetaPanda), "returns metapanda"
    # check dimensions
    assert result.p_ == df.shape[1] * 2 + 2, "concat should lead to twice the size of df for merge, +2 for indices"
    assert result.n_ == df.shape[0], "concat leads to same size"
    # no missing values
    assert result.df_.count().sum() == df.shape[0] * df.shape[1] * 2 + (df.shape[0] * 2), "missing values present"
