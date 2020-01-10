#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
from typing import Tuple, Union, Set, List, TypeVar, Callable, Optional, Dict

from numpy import ndarray
from pandas import pSeries, pDataFrame, pIndex
from matplotlib.pyplot import Figure
from .metapanda import MetaPanda
from .pipe import Pipe

# defined custom custypes.py
ArrayLike = Union[ndarray, pSeries, pDataFrame]
# index or series
PandaIndex = Union[pSeries, pIndex]
# set like
SetLike = Union[Set, pIndex, List, Tuple, pSeries]
# list or tuple
ListTup = Union[List, Tuple]
# a selector can be very broad.
SelectorType = Optional[Union[TypeVar, str, pIndex, Callable]]
# pipe for Pipelines
PipeTypeRawElem = Tuple[str, Tuple, Dict]
PipeTypeCleanElem = Tuple[Union[str, Callable, Dict, TypeVar], ...]
# specific pipe for MetaPanda
PipeMetaPandaType = Union[ListTup[PipeTypeCleanElem, ...], ListTup[PipeTypeRawElem, ...], str, Pipe]
# matplotlib Figure
MatPlotFig = Figure
# Broad DataSet Types
DataSetType = Union[pSeries, pDataFrame, MetaPanda]
