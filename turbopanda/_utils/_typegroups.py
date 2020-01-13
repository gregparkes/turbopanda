#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that define intuitive groupings of types."""

import numpy as np
from typing import TypeVar, Tuple


__all__ = ("c_float", "c_int", "t_numpy", "c_cat")


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
