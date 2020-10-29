#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for string functions."""

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

from turbopanda import str


def test_levenshtein():
    x = ["hello", "there", "I", "have", "the", "high", "ground"]
    y = ["you", "cannot", "stop", "my", "power"]

    zmat = str.levenshtein(x, as_matrix=True)