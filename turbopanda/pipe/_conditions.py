#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to conditions which can be directly fed into pandas.DataFrame.pipe."""
from __future__ import absolute_import, division, print_function
from typing import List
import pandas as pd


def select_float(x: pd.DataFrame) -> List[str]:
    """Only selects float columns."""
    return list(x.select_dtypes(include=["float"]).columns)


def select_numeric(x: pd.DataFrame) -> List[str]:
    """Only selects float or integer columns"""
    return list(x.select_dtypes(include=["float", "int"]).columns)
