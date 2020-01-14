#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles the dependencies within MetaPanda."""


def is_numpy_installed(raise_error: bool = False):
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
