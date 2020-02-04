#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles the dependencies within MetaPanda."""

__all__ = ('is_numpy_installed', 'is_scipy_installed', 'is_pandas_installed',
           'is_matplotlib_installed', 'is_sklearn_installed')


def is_numpy_installed(raise_error: bool = False):
    """Determines whether NumPy is installed."""
    try:
        import numpy  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("numpy needs to be installed. Please use `pip "
                      "install numpy`.")
    return is_installed


def is_scipy_installed(raise_error: bool = False):
    """Determines whether SciPy is installed."""
    try:
        import scipy  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("scipy needs to be installed. Please use `pip "
                      "install scipy`.")
    return is_installed


def is_pandas_installed(raise_error: bool = False):
    """Determines whether pandas is installed."""
    try:
        import pandas  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("pandas needs to be installed. Please use `pip "
                      "install pandas`.")
    return is_installed


def is_matplotlib_installed(raise_error: bool = False):
    """Determines whether matplotlib is installed."""
    try:
        import matplotlib.pyplot  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("matplotlib needs to be installed. Please use `pip "
                      "install matplotlib`.")
    return is_installed


def is_sklearn_installed(raise_error: bool = False):
    """Determines whether scikit-learn is installed."""
    try:
        import sklearn  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("sklearn needs to be installed. Please use `pip "
                      "install scikit-learn`.")
    return is_installed
