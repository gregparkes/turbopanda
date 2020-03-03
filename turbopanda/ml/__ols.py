#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct methods for ordinary least squares."""

import numpy as np


__all__ = ('direct_ols', 'direct_weighted_ols', 'direct_ridge')


def direct_ols(X, y):
    """Perform direct ordinary-least squares calculation if possible and well conditioned."""
    X = np.atleast_2d(np.asarray(X))
    y = np.atleast_1d(np.asarray(y))
    # compute and return betas
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def direct_weighted_ols(X, y):
    """Perform direct generalized ordinary-least squares calculation if possible and well conditioned."""
    X = np.atleast_2d(np.asarray(X))
    y = np.atleast_1d(np.asarray(y))
    # determine precision matrix as weighting to OLS
    P = np.linalg.pinv(np.cov(X))
    # compute this step independently as we don't want to recalculate twice.
    Xt_P = X.T @ P
    # compute and return betas
    return np.linalg.pinv(Xt_P @ X) @ Xt_P @ y


def direct_ridge(X, y, alpha=1.):
    """Perform direct ridge (l-2 normed OLS) calculation if possible and well conditioned."""
    X = np.atleast_2d(np.asarray(X))
    y = np.atleast_1d(np.asarray(y))
    return np.linalg.pinv(X.T @ X + alpha*np.eye(X.shape[1])) @ X.T @ y
