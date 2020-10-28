#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file handles the use of MetaPanda types."""

from typing import Callable, Optional, Type, Union

import pandas as pd

SelectorType = Optional[Union[Type, str, pd.Index, Callable]]
PandaIndex = Union[pd.Series, pd.Index]
