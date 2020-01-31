#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
from typing import Tuple, Union, Set, List, TypeVar, Callable, Optional, Dict, Any
from pandas import Series, DataFrame, Index

# index or series
PandaIndex = Union[Series, Index]
# a selector can be very broad.
SelectorType = Optional[Union[TypeVar, str, Index, Callable]]
# pipe for Pipelines
PipeTypeRawElem = Tuple[str, Tuple, Dict]
PipeTypeCleanElem = Tuple[Union[str, Callable, Dict, TypeVar], ...]
