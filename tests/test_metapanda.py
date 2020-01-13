#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for reading in files."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import turbopanda as turb


def test_read():
    x = turb.read("../data/translation.csv", name="Translation")
    assert isinstance(x, turb.MetaPanda)
