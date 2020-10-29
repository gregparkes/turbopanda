#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for utility functions."""
from __future__ import absolute_import, division, print_function

import math
import pytest
import os
import itertools as it

from hypothesis import given, example
import hypothesis.strategies as st
from hypothesis.extra import numpy, pandas

import numpy as np
import pandas as pd

from turbopanda import utils


@given(s=st.text())
def test_belongs1(s):
    l1 = ['apples', 'oranges', 'pears']
    with pytest.raises(ValueError):
        assert utils.belongs(s, l1)


def test_belongs2():
    l1 = ['apples', 'oranges', 'pears']
    assert utils.belongs("apples", l1)


def test_instance_check():
    # single example
    x = ['abba', 'father', 'cross']
    assert utils.instance_check(x, list)
    y = np.array([1, 2, 3], dtype=float)
    assert utils.instance_check(y, np.ndarray)
    # multiples given a tuple
    ij = True
    ji = False
    assert utils.instance_check((ij, ji), bool)


@given(st.integers(min_value=0))
def test_nonnegative1(x):
    assert utils.nonnegative(x, int)


@given(st.integers(max_value=-1))
def test_nonnegative2(x):
    with pytest.raises(AttributeError):
        assert utils.nonnegative(x, int)


def test_bounds_check():
    assert utils.bounds_check(math.pi, math.pi - 0.00001, math.pi + 0.00001)
    assert utils.bounds_check(5, 5 - 1, 5 + 1)


@given(numpy.arrays(float, (100, 2)), numpy.arrays(float, (100, 2)))
def test_arrays_equal_size(x, y):
    assert utils.arrays_equal_size(x, y)


@given(numpy.arrays(float, (100, 2)), st.text())
def test_arrays_dimension(array, s):
    with pytest.raises(ValueError):
        assert utils.arrays_dimension(array, s)


def test_check_list_type():
    x = [1, 2, 3]
    assert utils.check_list_type(x, int)
    y = ['a', 'b', 'c']
    assert utils.check_list_type(y, str)
    with pytest.raises(TypeError):
        assert utils.check_list_type(x, str)


@given(st.integers(min_value=1, max_value=150),
       st.sampled_from(["square", "diag"]),
       st.integers(min_value=1, max_value=15),
       st.integers(min_value=1, max_value=15),
       st.floats(min_value=0.01, max_value=10.))
def test_nearest_factors1(n, shape, cut, sr, wvar):
    # returns a tuple
    res = utils.nearest_factors(n, shape, cut, sr, wvar)

    assert type(res) == tuple
    assert isinstance(res[0], (np.int, np.int32, np.int64))
    assert isinstance(res[1], (np.int, np.int32, np.int64))

    diff = np.abs(res[0] * res[1] - sr)
    assert diff - 10 < diff < diff + 10


@given(st.lists(st.text(), min_size=1))
def test_zipe1(x):
    z = utils.zipe(x)

    assert type(z) == list
    assert len(z) == len(x)
    assert z[0] == x[0]


@given(st.lists(st.text(), min_size=1),
       st.lists(st.integers(), min_size=1))
def test_zipe2(x, y):
    z = utils.zipe(x, y)

    assert type(z) == list
    assert len(z) == max(len(x), len(y))
    assert z[0][0] == x[0]
    assert z[0][1] == y[0]


def test_zipe3():
    x = [1, 2, 3]
    assert utils.zipe(x) == x


def test_umap():
    """A mapping function"""

    def _f(x):
        """Blank."""
        return x ** 2

    assert utils.umap(_f, [2, 4, 6]) == [4, 16, 36]

    def _g(x, y):
        """Blank."""
        return x / y

    assert utils.umap(_g, [3, 6, 9], [1, 2, 3]) == [3, 3, 3]


def test_umapc():
    """mapping w cache"""

    def _f(x):
        """Blank."""
        return x ** 2

    assert utils.umapc("test.pkl", _f, [2, 4, 6]) == [4, 16, 36]
    # re-run to load from file
    assert utils.umapc("test.pkl", _f, [2, 4, 6]) == [4, 16, 36]
    assert os.path.isfile("test.pkl")
    os.remove("test.pkl")


def test_umapp():
    def _f(x):
        """Blank."""
        return x ** 2

    assert utils.umapp(_f, [2, 4, 6]) == [4, 16, 36]

    def _g(x, y):
        """Blank."""
        return x / y

    assert utils.umapp(_g, [3, 6, 9], [1, 2, 3]) == [3, 3, 3]


def test_umappc():
    """mapping w cache"""

    def _f(x):
        """Blank."""
        return x ** 2

    assert utils.umappc("test.pkl", _f, [2, 4, 6]) == [4, 16, 36]
    # re-run to load from file
    assert utils.umappc("test.pkl", _f, [2, 4, 6]) == [4, 16, 36]
    assert os.path.isfile("test.pkl")
    os.remove("test.pkl")

    def _g(x, y):
        """Blank."""
        return x / y

    assert utils.umappc("test2.pkl", _g, [3, 6, 9], [1, 2, 3]) == [3, 3, 3]
    assert utils.umappc("test2.pkl", _g, [3, 6, 9], [1, 2, 3]) == [3, 3, 3]
    assert os.path.isfile("test2.pkl")
    os.remove("test2.pkl")


@given(pandas.series(dtype=int))
def test_panderfy(ser):
    # make a copy
    cpy = ser.copy()
    pd.testing.assert_series_equal(ser, utils.transform_copy(ser, cpy))


@given(pandas.indexes(dtype=int))
def test_panderfy2(ser):
    # make a copy
    cpy = ser.copy()
    pd.testing.assert_index_equal(ser, utils.transform_copy(ser, cpy))


@given(pandas.data_frames(
    [pandas.column("A", dtype=int),
     pandas.column("B", dtype=float)
     ]))
def test_panderfy3(ser):
    # make a copy
    cpy = ser.copy()
    pd.testing.assert_frame_equal(ser, utils.transform_copy(ser, cpy))


@given(numpy.arrays(int, (100,)))
def test_remove_na(x):
    np.testing.assert_array_almost_equal(x, utils.remove_na(x))


def test_dict_to_tuple():
    d = {"a": 1, "b": 2, "c": 3}
    assert utils.dict_to_tuple(d) == (("a", 1), ("b", 2), ("c", 3))


def test_dictsplit():
    d = {"a": 1, "b": 2, "c": 3}
    assert utils.dictsplit(d) == (tuple(d.keys()), tuple(d.values()))


def test_dictzip():
    x = [1, 2, 3]
    y = ['a', 'b', 'c']
    assert utils.dictzip(x, y) == dict(zip(x, y))


@given(st.text())
def test_set_like(x):
    assert pd.testing.assert_index_equal(pd.Index([x]), utils.set_like(x))


@given(st.lists(st.text()))
def test_set_like2(x):
    assert pd.testing.assert_index_equal(pd.Index(sorted(set(x))), utils.set_like(x))


@given(st.tuples(st.text()))
def test_set_like3(x):
    assert pd.testing.assert_index_equal(pd.Index(sorted(set(x))), utils.set_like(x))


@given(pandas.series(dtype=str))
def test_set_like4(x):
    assert pd.testing.assert_index_equal(pd.Index(x.dropna().unique()), utils.set_like(x))


@given(st.sets(st.text()))
def test_set_like5(x):
    assert pd.testing.assert_index_equal(pd.Index(set(x)), utils.set_like(x))


@given(pandas.indexes(dtype=str))
def test_set_like6(x):
    assert pd.testing.assert_index_equal(pd.Index(x.dropna().unique()), utils.set_like(x))


def test_union():
    x = ["fi", "fo", "fum"]
    y = ["fi", "yo", "sum"]
    z = ["fi", "fe", "sun"]

    assert np.all(utils.union(x, y) == pd.Index(["fi", "fo", "fum", "sum", "yo"]))
    assert np.all(utils.union(x, y, z) == pd.Index(["fe", "fi", "fo", "fum", "sum", "sun", "yo"]))


def test_intersect():
    x = ["fi", "fo", "sun"]
    y = ["fi", "yo", "sum"]
    z = ["fi", "fe", "sun"]

    assert np.all(utils.intersect(x, z) == pd.Index(["fi", "sun"]))
    assert np.all(utils.intersect(x, y, z) == pd.Index(["fi"]))


def test_difference():
    x = ["fi", "fo", "sun"]
    z = ["fi", "yo", "sum"]

    assert np.all(utils.difference(x, z) == pd.Index(["fo", "sum", "sun", "yo"]))


def test_absdifference():
    x = ["fi", "fo", "sun"]
    z = ["fi", "yo", "sum"]

    assert np.all(utils.absdifference(x, z) == pd.Index(["fo", "sun"]))


@given(f=st.functions(like=lambda i, j: i + j, returns=st.integers()),
       x=st.lists(st.integers(), max_size=10))
def test_pairwise(f, x):
    r = utils.pairwise(f, x)
    assert type(r) == list
    assert len(r) == len(list(it.combinations(x, 2)))


