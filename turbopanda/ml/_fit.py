#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.base import is_classifier, is_regressor

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import listify, union, standardize, instance_check


def cleaned_subset(df, x, y):
    """Determines an optimal subset with no missing values.

    Parameters
    ----------
    df : MetaPanda
    x : selector
    y : str

    Returns
    -------

    """
    _x = df.view(x)
    cols = union(_x, [y])
    return df[cols].dropna()


def _find_sklearn_model(name):
    if isinstance(name, str):
        packages = [
            sklearn.linear_model, sklearn.tree, sklearn.neighbors
        ]
        for pkg in packages:
            if hasattr(pkg, name):
                return getattr(pkg, name)(), pkg.__name__
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))
    elif is_classifier(name):
        return name
    elif is_regressor(name):
        return name
    else:
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


def _extract_coefficients_from_model(cv, x, pkg_name):
    if pkg_name == "sklearn.linear_model":
        cof = np.vstack([m.coef_ for m in cv['estimator']])
        if cof.shape[-1] == 1:
            cof = cof.flatten()
        res = pd.DataFrame(cof, columns=listify(x))
        res['intercept'] = np.vstack([m.intercept_ for m in cv['estimator']]).flatten()
        res.columns = union(listify(x), ['intercept'])
        return res
    elif pkg_name == "sklearn.tree" or pkg_name == "sklearn.ensemble":
        cof = np.vstack([m.feature_importances_ for m in cv['estimator']])
        if cof.shape[-1] == 1:
            cof = cof.flatten()
        res = pd.DataFrame(cof, columns=listify(x))
        res.columns = pd.Index(listify(x))
        return res
    else:
        return []


def fit_basic(df: MetaPanda,
              x: SelectorType,
              y: str,
              k: int = 5,
              repeats: int = 100,
              model: str = "LinearRegression"):
    """Performs a rudimentary fit model with no parameter searching.

    Parameters
    ----------
    df : MetaPanda
        The main dataset.
    x : list/tuple of str
        A list of selected column names for x or MetaPanda `selector`.
    y : str
        A selected y column.
    k : int, optional
        The number of cross-fold validations
    repeats : int, optional
        We use RepeatedKFold, so specifying some repeats
    model : str, sklearn model
        The name of a scikit-learn model, or the model object itself.

    Returns
    -------
    cv : pd.DataFrame
        A dataframe result of cross-validated repeats. Can include w_ coefficients.
    yp : np.ndarray
        The predictions for each of y
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(k, int)
    instance_check(repeats, int)

    # define sk model.
    rep = RepeatedKFold(n_splits=k, n_repeats=repeats)
    lm, pkg_name = _find_sklearn_model(model)
    # use view to select x columns.
    xcols = df.view(x)
    # join together and fetch subset to drop
    cols = union(xcols, y)
    _df = df[cols].dropna()

    # split into x and y
    _x = np.asarray(_df[xcols]).reshape(-1, 1) if len(x) == 1 else np.asarray(_df[xcols])
    _y = np.asarray(_df[y])

    # standardize
    _x = standardize(_x)

    # cross validate and predict values.
    cv = cross_validate(lm, _x, _y, cv=rep, return_estimator=True, return_train_score=True)
    yp = cross_val_predict(lm, _x, _y, cv=k)

    # extract coefficients.
    coef = _extract_coefficients_from_model(cv, xcols, pkg_name)

    results = pd.DataFrame(cv)
    # add extra columns for utility.
    results['k'] = np.repeat(np.arange(k), repeats)
    # integrate coefficients into cv if present.
    if not isinstance(coef, (list, tuple)):
        # join on coefficients, prefixing with 'w__' for weights.
        results = results.join(coef.add_prefix("w__"))
    # drop 'estimator' objects.
    results.drop('estimator', axis=1, inplace=True)

    # make it a metapanda for ease.
    return MetaPanda(results), yp
