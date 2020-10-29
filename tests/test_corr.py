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

from scipy.stats import pearsonr

import turbopanda as turb


class TestCorrModule(unittest.TestCase):

    def test_bicorr(self):
        # logic check
        t1 = pd.Series([1., 0., 0., 1.])
        t2 = pd.Series([1., 0., 0., 1.])
        # correlation should be 1
        self.assertAlmostEqual(turb.corr.bicorr(t1, t2), 1.0)

    @given(x=pandas.series(elements=st.floats()),
           y=pandas.series(elements=st.floats()))
    def test_bicorr2(self, x, y):
        self.assertAlmostEqual(turb.corr.bicorr(x, y), pearsonr(x, y)[0])
