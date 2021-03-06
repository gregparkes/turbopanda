#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that define intuitive groupings of types."""

from typing import Any, Tuple, Type

import numpy as np
from pandas import Series, DataFrame, Index

__all__ = ("c_float", "c_int", "t_numpy", "c_cat", "fself")


def fself(x: Any) -> Any:
    """Returns itself."""
    return x


def c_float() -> Tuple[
    Type[np.float_], Type[np.single], Type[np.half], Type[float], Type[float]
]:
    """Returns accepted float types."""
    return np.float64, np.float32, np.float16, np.float, float


def c_int() -> Tuple[
    Type[np.long],
    Type[np.intc],
    Type[np.short],
    Type[np.int8],
    Type[np.int],
    Type[np.uintp],
    Type[np.uint8],
    Type[np.ushort],
    Type[np.ushort],
    Type[np.uintc],
    Type[np.int],
]:
    """Returns accepted integer types."""
    return (
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.int,
        np.uint,
        np.uint8,
        np.uint16,
        np.uint16,
        np.uint32,
        int,
    )


def t_numpy() -> Tuple[
    Type[np.int],
    Type[np.bool],
    Type[np.float],
    Type[np.float_],
    Type[np.single],
    Type[np.half],
    Type[np.long],
    Type[np.intc],
    Type[np.short],
    Type[np.int8],
    Type[np.uint8],
    Type[np.ushort],
    Type[np.uintc],
    Type[np.uintp],
]:
    """Returns the supported numeric types from NumPy."""
    return (
        np.int,
        np.bool,
        np.float,
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )


def c_cat() -> Tuple[Type[np.uint8], Type[np.ushort], Type[object]]:
    """Returns accepted category types."""
    return np.uint8, np.uint16, object


""" Miscallaenous type groups to use for instance checks """


def one_dim() -> Tuple[
    Type[list], Type[tuple], Type[Series], Type[Index], Type[np.ndarray]
]:
    """Returns 1D-like elements"""
    return list, tuple, Series, Index, np.ndarray


def two_dim() -> Tuple[Type[np.ndarray], Type[DataFrame]]:
    """Returns 2D-like elements"""
    return np.ndarray, DataFrame
