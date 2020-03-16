#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods for doing checks on numpy arrays."""


import numpy as np


__all__ = ('is_float_array', 'is_int_array')


def is_float_array(ndarr) -> bool:
    """Determines whether `ndarr` is an array of floats."""
    return ndarr.dtype.kind == 'f'


def is_int_array(ndarr) -> bool:
    """Determines whether `ndarr` is an array of integers"""
    return ndarr.dtype.kind == 'i'
