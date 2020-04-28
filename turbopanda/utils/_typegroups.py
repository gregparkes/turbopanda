#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that define intuitive groupings of types."""

from typing import Any, Tuple, TypeVar

import numpy as np
import pandas as pd

__all__ = ("c_float", "c_int", "t_numpy", "c_cat", 'fself')


def fself(x: Any) -> Any:
    """Returns itself."""
    return x


def c_float() -> Tuple[TypeVar, ...]:
    """Returns accepted float types."""
    return np.float64, np.float32, np.float16, np.float, float


def c_int() -> Tuple[TypeVar, ...]:
    """Returns accepted integer types."""
    return (np.int64, np.int32, np.int16, np.int8,
            np.int, np.uint, np.uint8, np.uint16,
            np.uint16, np.uint32, int)


def t_numpy() -> Tuple[TypeVar, ...]:
    """Returns the supported numeric types from NumPy."""
    return (
        np.int, np.bool, np.float, np.float64, np.float32, np.float16, np.int64,
        np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64
    )


def c_cat() -> Tuple[TypeVar, ...]:
    """Returns accepted category types."""
    return np.uint8, np.uint16, object


""" Miscallaenous type groups to use for instance checks """


def one_dim() -> Tuple[TypeVar, ...]:
    """Returns 1D-like elements"""
    return list, tuple, pd.Series, pd.Index, np.ndarray


def two_dim() -> Tuple[TypeVar, ...]:
    """Returns 2D-like elements"""
    return np.ndarray, pd.DataFrame
