#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd
from typing import Optional, List

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.base import is_classifier, is_regressor

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import listify, union, standardize, instance_check
from ._plot import overview_plot
from ._clean import ml_ready


def _save_cache_basic_fit(cache_name: str,
                          df: MetaPanda,
                          x: List,
                          y: str,
                          k: int,
                          model: str,
                          cv: MetaPanda,
                          yp: pd.Series):
    """Given the information from basic_fit, cache the data into a resultfile.

    Parameters in JSON:
        type: basic
        version: sklearn: 0.2.1, numpy: x, pandas: y, metapanda: z...
        model: `LinearRegression`
        package: `sklearn.linear_model`
        x: <list of column names>
        y: <column name>
        n: number of samples
        cv: results matrix
        yp: fitted values
        source: dataframe source containing x, y, as filename absolute path
    """
    import os
    import sklearn
    import turbopanda as turb
    import hashlib
    package = _find_sklearn_package(model)

    # calculate a checksum based on model:source:x:y:n:k:cv
    cv_digest = hashlib.sha256(cv.df_.to_json().encode()).hexdigest()
    model_digest = hashlib.sha256(model.encode()).hexdigest()
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    x_digest = hashlib.sha256(json.dumps(df.view(x).tolist()).encode()).hexdigest()
    y_digest = hashlib.sha256(y.encode()).hexdigest()
    n_digest = hashlib.sha256(str(yp.shape[0]).encode()).hexdigest()
    k_digest = hashlib.sha256(str(k).encode()).hexdigest()
    # create length check sum string
    chk = model_digest + source_digest + x_digest + y_digest + n_digest + k_digest + cv_digest

    js = json.dumps({
        'type': 'basic',
        'version': {
            'numpy': np.__version__, 'pandas': pd.__version__,
            'sklearn': sklearn.__version__, 'turbopanda': turb.__version__
        },
        'model': model,
        'package': package,
        'source': df.source_,
        'x': df.view(x).tolist(),
        'y': y,
        'n': yp.shape[0],
        'k': str(k),
        'cv': cv.df_.to_dict(),
        'yp': yp.to_dict(),
        'chk': chk
    })

    if cache_name:
        with open(cache_name, "w") as f:
            json.dump(js, f)


def _find_sklearn_package(name):
    if isinstance(name, str):
        packages = [
            sklearn.linear_model, sklearn.tree, sklearn.neighbors
        ]
        for pkg in packages:
            if hasattr(pkg, name):
                return pkg.__name__
    raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


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
              model: str = "LinearRegression",
              cache: Optional[str] = None,
              plot: bool = False):
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
    cache : str, optional
        If not None, stores the resulting model parts in JSON and reloads if present.
    plot : bool, optional
        If True, produces `overview_plot` inplace.

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
    instance_check(cache, (type(None), str))

    import os

    if cache is not None:
        # see if file is present, and if so, load.
        if os.path.isfile(cache):
            # load
            pass

    # define sk model.
    rep = RepeatedKFold(n_splits=k, n_repeats=repeats)
    lm, pkg_name = _find_sklearn_model(model)

    # use view to select x columns.
    xcols = df.view(x)
    # get ml ready versions
    _df, _x, _y = ml_ready(df, x, y)

    # cross validate and predict values.
    cv = cross_validate(lm, _x, _y, cv=rep, return_estimator=True, return_train_score=True, n_jobs=-2)
    yp = cross_val_predict(lm, _x, _y, cv=k)

    # compute direct methods if they exist
    """
    if model == "LinearRegression":
        _beta = _direct_ols(_x, _y)
    elif model == 'GeneralizedLeastSquares':
        _beta = _direct_weighted_ols(_x, _y)
    elif model == 'Ridge':
        _beta = _direct_ridge(_x, _y, 1.)
    """

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

    _cv = MetaPanda(results)
    _yp = pd.Series(yp, index=_df.index)

    if plot:
        overview_plot(df, x, y, _cv, _yp)

    # make it a metapanda for ease.
    return _cv, _yp
