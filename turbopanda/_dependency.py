#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles the dependencies within MetaPanda."""

import functools
from turbopanda.utils import belongs

__all__ = (
    "is_numpy_installed",
    "is_scipy_installed",
    "is_pandas_installed",
    "is_matplotlib_installed",
    "is_sklearn_installed",
    "is_numba_installed",
    "is_tqdm_installed",
    "is_difflib_installed",
    "requires"
)


def is_numpy_installed(raise_error: bool = False):
    """Determines whether NumPy is installed."""
    try:
        import numpy as np  # noqa

        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("numpy not installed. Use `pip install numpy`.")
    return is_installed


def is_numba_installed(raise_error: bool = False):
    """Determines whether NumPy is installed."""
    try:
        import numba  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("numba not installed. Use `pip install numba`.")
    return is_installed


def is_scipy_installed(raise_error: bool = False):
    """Determines whether SciPy is installed."""
    try:
        import scipy  # noqa

        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("scipy not installed. Use `pip " "install scipy`.")
    return is_installed


def is_pandas_installed(raise_error: bool = False):
    """Determines whether pandas is installed."""
    try:
        import pandas as pd  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("pandas not installed. Use `pip install pandas`.")
    return is_installed


def is_matplotlib_installed(raise_error: bool = False):
    """Determines whether matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("matplotlib not installed. Use `pip install matplotlib`.")
    return is_installed


def is_sklearn_installed(raise_error: bool = False):
    """Determines whether scikit-learn is installed."""
    try:
        import sklearn  # noqa

        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("sklearn not installed. Use `pip install scikit-learn`.")
    return is_installed


def is_joblib_installed(raise_error: bool = False):
    """Determines whether joblib is installed or not."""
    try:
        import joblib  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("joblib not installed. Use `pip install joblib`.")
    return is_installed


def is_tqdm_installed(raise_error: bool = False):
    """Determines whether SciPy is installed."""
    try:
        from tqdm import tqdm  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("tqdm not installed. Use `pip " "install tqdm`.")
    return is_installed


def is_difflib_installed(raise_error: bool = False):
    """Determines whether SciPy is installed."""
    try:
        import difflib  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("difflib not installed. Use `pip " "install difflib`.")
    return is_installed


def requires(*package_names: str):
    """A decorator for requiring certain packages.

        Parameters
        ----------
        package_names : str args
            Arguments The name of the package needed to run this program
        """
    packages = ('numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn',
                'joblib', 'tqdm', 'numba', 'difflib')
    aligned_funcs = (is_numpy_installed, is_scipy_installed, is_matplotlib_installed,
                     is_pandas_installed,
                     is_sklearn_installed, is_joblib_installed,
                     is_tqdm_installed, is_numba_installed,
                     is_difflib_installed)
    # ensure package_name is one of the selected.
    hash_pack = dict(zip(packages, aligned_funcs))

    def _decorator_requires(func):
        @functools.wraps(func)
        def _inner_func(*args, **kwargs):
            # call the appropriate checking function
            for pn in package_names:
                hash_pack[pn](True)
            # now return as normal
            return func(*args, **kwargs)

        return _inner_func

    return _decorator_requires
