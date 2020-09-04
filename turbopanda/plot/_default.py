#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for different defaults to use in conjunction with Matplotlib."""

import matplotlib as mpl
from typing import List, Tuple, Optional, Union
from numpy import ndarray
from pandas import Series

# define a bunch of plot types
_Numeric = Union[int, float]
_ListLike = Union[List, Tuple, ndarray]
_ArrayLike = Union[List, Tuple, ndarray, Series]
