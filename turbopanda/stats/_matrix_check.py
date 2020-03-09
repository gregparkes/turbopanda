#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic methods for checking matrices."""

import numpy as np


def is_invertible(X):
    """Check whether matrix X is invertible."""
    return X.shape[0] == X.shape[1] and np.linalg.matrix_rank(X) == X.shape[0]
