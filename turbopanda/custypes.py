#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
from typing import Tuple, Union, Set, List, TypeVar, Callable, Optional, Dict, Any
from numpy import ndarray
from pandas import Series, DataFrame, Index
from matplotlib.pyplot import Figure

# defined custom custypes.py
ArrayLike = Union[ndarray, Series, DataFrame]
# index or series
PandaIndex = Union[Series, Index]
# set like
SetLike = Union[type(None), str, Set, Index, List, Tuple, Series]
# a selector can be very broad.
SelectorType = Optional[Union[TypeVar, str, Index, Callable]]
# pipe for Pipelines
PipeTypeRawElem = Tuple[str, Tuple, Dict]
PipeTypeCleanElem = Tuple[Union[str, Callable, Dict, TypeVar], ...]
# matplotlib Figure
MatPlotFig = Figure
