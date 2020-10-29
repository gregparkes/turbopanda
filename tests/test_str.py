#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for string functions."""

from __future__ import absolute_import, division, print_function

import math
import pytest
import os
import re
import itertools as it

from hypothesis import given, example, note, assume
import hypothesis.strategies as st
from hypothesis.extra import numpy, pandas

import unittest

import numpy as np
import pandas as pd

import turbopanda as turb


random_or_regex = st.one_of(st.text(min_size=1), st.from_regex("ellow"))


def int_dataframe_example():
    """

    Returns
    -------

    """
    return pd.DataFrame({'hello': [1, 2, 3], 'mellow': [4, 5, 6],
                         'bellow': [7, 8, 9], 'swellow': ['fi', 'fe', 'fo']})


def str_dataframe_example():
    """

    Returns
    -------

    """
    return pd.DataFrame({'hello': ["carrots", "apples", "pears"],
                         'mellow': [4, 5, 6],
                         'bellow': ["bird", "shark", "dragon"],
                         'swellow': ['fi', 'fe', 'fo']})


class TestStringModule(unittest.TestCase):

    def test_common_substrings(self):
        t = ['hello there!', 'hey there', 'well hello there', 'there is a disturbance']

        x1 = turb.str.common_substrings("high tower", "low tower")
        assert type(x1) == np.str
        assert x1 == " tower"

        x2 = turb.str.common_substrings(t)
        # should return a series
        assert type(x2) == pd.Series
        assert len(x2) == 2
        assert x2.index[0] == 'there'
        assert x2.iloc[0] == 3

    @given(s1=st.one_of(
        st.text(), st.lists(st.text())),
        s2=st.one_of(
            st.text(), st.lists(st.text())))
    def test_common_substrings2(self, s1, s2):
        x1 = turb.str.common_substrings(s1, s2)
        note(x1)
        if type(s1) == str and type(s2) == str:
            assert type(x1) == str
        else:
            assert type(x1) == pd.Series

    def test_pattern(self):
        _in = ["hello", "bellow", "mellow", "swellow"]
        x1 = turb.str.pattern("ellow", _in)
        assert type(x1) == pd.Index
        assert len(x1) == 3
        assert np.all(x1, pd.Index(['bellow', 'mellow', 'swellow']))

        x2 = turb.str.pattern("^he | ^b", _in)
        assert type(x2) == pd.Index
        assert len(x2) == 2
        assert np.all(x2, pd.Index(['hello', 'bellow']))

        _in2 = int_dataframe_example()
        x3 = turb.str.pattern("^he | ^b", _in2)
        assert type(x3) == pd.Index
        assert len(x3) == 2
        assert np.all(x3, pd.Index(['hello', 'bellow']))

    @given(words=st.one_of(
        st.lists(random_or_regex, unique=True),
        pandas.series(elements=random_or_regex, dtype=str, unique=True),
        pandas.indexes(elements=random_or_regex, dtype=str, unique=True),
    ))
    def test_pattern2(self, words):
        res = list(turb.str.pattern("ellow", words))
        # select all that conform to the pattern..?
        actual = [x for x in words if re.search("ellow", x)]
        self.assertEqual(res, actual)

    def test_patproduct(self):
        x1 = turb.str.patproduct("%s%d", ("x", "y"), range(100))
        assert type(x1) == list
        assert len(x1) == 200
        assert x1[0] == 'x0'
        assert x1[-1] == 'y99'

        # second example
        x2 = turb.str.patproduct("%s_%s", ("repl", "quality"), ("sum", "prod"))
        assert type(x2) == list
        assert len(x2) == 4
        assert x2[0] == 'repl_sum'
        assert x2[-1] == "quality_prod"

    @given(a=st.tuples(st.text(min_size=1), st.text(min_size=1)),
           b=st.lists(st.text(min_size=1)))
    def test_patproduct2(self, a, b):
        x1 = turb.str.patproduct("%s%s", a, b)
        self.assertListEqual(x1, ["%s%s" % item for item in it.product(a, b)])

    @given(s=st.text(max_size=50),
           strat=st.sampled_from(['middle', 'end']))
    def test_shorten(self, s, strat):
        ns = turb.str.shorten(s, strategy=strat)
        note(ns)
        assert type(ns) == np.str
        if len(s) < 15:
            assert len(s) == len(ns)
        else:
            assert len(ns) <= 15

    @given(st.lists(st.text(max_size=70)))
    def test_shorten2(self, ls):
        nls = turb.str.shorten(ls)
        assert type(nls) == list

    def test_string_replace(self):
        t = ['hello', 'i am', 'pleased']
        r = ['hello', 'u am', 'pleased']
        x1 = turb.str.string_replace(t, ('i', 'u'))
        assert type(x1) == list
        assert len(x1) == 3
        self.assertEqual(x1, r)

    @given(st.text(min_size=1))
    def test_string_replace2(self, s):
        op1 = ('u', 'i')
        res = s.replace(*op1)
        assert turb.str.string_replace(s, op1) == res

    @given(st.lists(st.text(min_size=1), min_size=1))
    def test_string_replace3(self, s):
        ops = [('u', 'i'), ('a', 'e')]
        res = s[0].replace(*ops[0]).replace(*ops[1])
        assert turb.str.string_replace(s, *ops)[0] == res

    def test_reformat(self):
        df = str_dataframe_example()
        x1 = turb.str.reformat("{hello}{bellow}", df)

        assert type(x1) == pd.Series
        pd.testing.assert_series_equal(x1, pd.Series(['carrotsbird', "applesshark", "pearsdragon"], dtype=object))

    @given(df=pandas.data_frames([pandas.column("A", st.text(min_size=1),
                                                dtype=str, unique=True),
                                  pandas.column("B", st.text(min_size=1),
                                                dtype=str, unique=True)]))
    def test_reformat2(self, df):
        x1 = turb.str.reformat("{A}{B}", df)
        note(x1)
        pd.testing.assert_series_equal(x1, df['A'] + df['B'])
