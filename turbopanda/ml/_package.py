#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles package-based methods for sklearn."""

import sklearn
from sklearn.base import is_classifier, is_regressor
from turbopanda.utils import panderfy

__all__ = ('skpackages', 'is_sklearn_model',
           'find_sklearn_package', 'find_model_family', 'find_sklearn_model')


def skpackages():
    """Returns compliant sklearn packages."""
    return (sklearn.linear_model, sklearn.tree, sklearn.neighbors,
            sklearn.ensemble, sklearn.svm, sklearn.gaussian_process)


def is_sklearn_model(name):
    """Returns whether `name` is a valid sklearn model."""
    if isinstance(name, str):
        for pkg in skpackages():
            if hasattr(pkg, name):
                return True
    elif is_classifier(name):
        return True
    elif is_regressor(name):
        return True
    else:
        return False
    return False


def find_sklearn_package(name):
    """Given string name, find the package associated with sklearn model."""
    if isinstance(name, str):
        for pkg in skpackages():
            if hasattr(pkg, name):
                return pkg.__name__
    raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


def find_sklearn_model(name):
    """Given string name, find the sklearn object and module associated."""
    if isinstance(name, str):
        for pkg in skpackages():
            if hasattr(pkg, name):
                return getattr(pkg, name)(), pkg.__name__
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))
    elif is_classifier(name):
        return name, name.__module__.rsplit(".", 1)[0]
    elif is_regressor(name):
        return name, name.__module__.rsplit(".", 1)[0]
    else:
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


@panderfy
def find_model_family(models):
    """Given a list of model names, return the sklearn family package name it belongs to."""
    return [find_sklearn_package(m).split(".")[-1] for m in models]
