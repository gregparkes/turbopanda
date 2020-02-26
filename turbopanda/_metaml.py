#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:01:53 2019

@author: gparkes

Creates Meta Scikit-learn models.
"""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, gaussian_process, neighbors, tree, svm
from sklearn.base import is_classifier, is_regressor, BaseEstimator
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import r2_score

# locals
from ._pipe import Pipe
from .ml._default import param_types, model_types
from ._metapanda import MetaPanda


__sklearn_model_packages__ = (tree, linear_model, ensemble, svm, gaussian_process, neighbors)


def _get_hidden_coefficients(directory):
    return [d for d in directory if d.endswith("_") and d.islower()]


def _is_multioutput_wrap(model):
    return isinstance(model, (MultiOutputRegressor, MultiOutputClassifier))


def _multioutput_wrap(model_t):
    if is_classifier(model_t):
        return MultiOutputClassifier(model_t)
    elif is_regressor(model_t):
        return MultiOutputRegressor(model_t)


def _find_sklearn_model(model_str):
    if isinstance(model_str, str):
        for package in __sklearn_model_packages__:
            if hasattr(package, model_str):
                return getattr(package, model_str)
    elif isinstance(model_str, BaseEstimator):
        return model_str
    else:
        raise ImportError("model '{}' not found in any sklearn.package.".format(model_str))


def _wrap_pandas(data, labels):
    if isinstance(labels, str) and data.ndim == 1:
        return pd.Series(data, name=labels)
    elif data.ndim == 1:
        return pd.Series(data, index=labels)
    else:
        return pd.DataFrame(data, columns=labels)


def _coefficient_multioutput_block(fitted, X, y):
    if hasattr(fitted[0], "coef_") and hasattr(fitted[0], "intercept_"):
        cols = X.columns if isinstance(X, pd.DataFrame) else [X.name]
        _coef_mat = pd.concat([
            pd.DataFrame([mf.intercept_ for mf in fitted], columns=["intercept"]).T,
            pd.DataFrame(np.vstack(([mf.coef_ for mf in fitted])).T, index=cols)
        ])
        _coef_mat.columns = y.columns
        return _coef_mat
    elif hasattr(fitted[0], "feature_importances_"):
        cols = X.columns if isinstance(X, pd.DataFrame) else [X.name]
        _coef_mat = pd.concat([
            pd.DataFrame(np.vstack(([mf.feature_importances_ for mf in fitted])).T, index=cols)
        ])
        _coef_mat.columns = y.columns
        return _coef_mat
    else:
        return []


def _coefficient_multioutput(fitted, X, y, cv):
    # generate blocks, where each block is a CV
    blocks = [_coefficient_multioutput_block(b.estimators_, X, y) for b in fitted]
    nd = pd.concat(blocks, axis=1)
    # create multicolumn
    z_t = zip(nd.columns, np.repeat(np.arange(0, cv, 1, dtype=int), y.columns.shape[0]))
    nd.columns = pd.MultiIndex.from_tuples(z_t, names=("cv", "y"))
    return nd


def _get_coefficient_normal(fitted_models, X):
    # check if wrapped in multioutput first
    if hasattr(fitted_models[0], "coef_") and hasattr(fitted_models[0], "intercept_"):
        # linear model or SVM
        # coefficient matrix
        cols = X.columns if isinstance(X, pd.DataFrame) else [X.name]
        _coef_mat = pd.concat([
            pd.DataFrame([mf.intercept_ for mf in fitted_models], columns=["intercept"]).T,
            pd.DataFrame(np.vstack(([mf.coef_ for mf in fitted_models])).T, index=cols)
        ])
        _coef_mat.columns.name = "cv"
        return _coef_mat
    elif hasattr(fitted_models[0], "feature_importances_"):
        # tree based model
        # coef matrix
        cols = X.columns if isinstance(X, pd.DataFrame) else [X.name]
        _coef_mat = pd.concat([
            pd.DataFrame(np.vstack(([mf.feature_importances_ for mf in fitted_models])).T, index=cols)
        ])
        _coef_mat.columns.name = "cv"
        return _coef_mat
    else:
        return None


def _get_coefficient_matrix_multioutput(fitted_multi, X, y, cv):
    if _is_multioutput_wrap(fitted_multi[0]):
        t_X = pd.concat([
            pd.DataFrame([
                e.coef_ for e in m.estimators_
            ], columns=X.columns, index=y.columns).T
            for m in fitted_multi], axis=1
        )
        # assign multiindex
        z_t = zip(
            np.repeat(np.arange(0, cv, 1, dtype=np.int), y.columns.shape[0]), X.columns
        )
        n_col = pd.MultiIndex.from_tuples(z_t, names=("cv", "y"))
        t_X.columns = n_col


class MetaML(object):
    """Performs complex machine-learning operations on a MetaPanda object to run
    scikit-learn models.

    Attributes
    ----------
    cv : int
        Number of cross-validations
    X : pd.DataFrame
        The input columns
    y : pd.DataFrame
        The output column(s)
    model_str : str
        The name of the scikit-learn model to use
    mdf : MetaPanda
        The entire dataset to call upon
    lm : sklearn model
        The actual model to perform analyses on
    is_fit : bool
        Determines whether the model has been fitted or not
    x_names : list
        The column names in X
    y_names : str/list
        The column name(s) in y
    score_r2 : float
        The score of the fitted model
    coef_mat : pd.DataFrame
        The result of model coefficients from the fitted model

    Methods
    -------
    fit()
        Performs a single run/fit and returns predictions and scores based on defaults.
    """

    def __init__(self, mp: MetaPanda, x_select, y_select, model: str = "LinearRegression"):
        """
        Receives a MetaPanda object with all of the features ready to go(ish).

        Parameters
        --------
        mp : MetaPanda
            A pandas.DataFrame object containing preprocessed input/output
        x_select : selector
            A selector of x-inputs
        y_select : selector
            A selector of y-inputs
        model : str, sklearn model, optional
            The model selected to perform a run as
        """
        # make mp ML-ready
        self.model_str = model
        # compute ML pipeline to dataset
        self.mdf_ = mp.compute(Pipe.ml_regression(mp, x_select, y_select), inplace=False)
        # call cache x and y selectors
        self.mdf_.cache("input", x_select)
        self.mdf_.cache("output", y_select)
        # create X and y
        self.X, self.y = self.mdf_[x_select], self.mdf_[y_select]
        # defaults
        self.cv = 10
        # find and instantiate
        model_t = _find_sklearn_model(model)()
        # set default parameter of primary parameter
        _mt = model_types()
        _pt = param_types()

        prim_param = _mt.loc[model, "Primary Parameter"]
        if prim_param is not np.nan:
            def_value = _pt.loc[prim_param, "Default"]
            # set parameter of primary value
            model_t.set_params(**{prim_param: def_value})

        # cover for multioutput
        if self.multioutput:
            self.lm = _multioutput_wrap(model_t)
        else:
            self.lm = model_t

        self.is_fit = False

    """ ################################ PROPERTIES #############################################"""

    @property
    def cv(self) -> int:
        """The number of cross-validations."""
        return self._cv

    @cv.setter
    def cv(self, _cv):
        if isinstance(_cv, (int, np.int)):
            self._cv = _cv
        else:
            raise TypeError("cv is not of type [int]")

    @property
    def X(self) -> pd.DataFrame:
        """The input matrix into the ML model."""
        return self._X

    @X.setter
    def X(self, _x):
        if isinstance(_x, (pd.DataFrame, pd.Series)):
            self._X = _x
        else:
            raise TypeError("X input is not of type [pd.DataFrame, pd.Series]")

    @property
    def y(self):
        """The target vector of the ML model."""
        return self._y

    @y.setter
    def y(self, _y):
        if isinstance(_y, (pd.DataFrame, pd.Series)):
            self._y = _y
        else:
            raise TypeError("y output is not of type [pd.DataFrame, pd.Series]")

    @property
    def model_str(self) -> str:
        """The string representation of the sklearn model."""
        return self._model_str

    @model_str.setter
    def model_str(self, ms):
        self._model_str = ms

    @property
    def x_names(self):
        """The x column names."""
        return self.X.columns if self.X.ndim > 1 else [self.X.name]

    @property
    def _x_numpy(self):
        """The numpy representation of input x."""
        return self.X.values.reshape(-1, 1) if self.X.ndim == 1 else self.X.values

    @property
    def y_names(self):
        """The y column names."""
        return self.y.columns if self.multioutput else self.y.name

    @property
    def multioutput(self) -> bool:
        """Whether the data is multioutput or not."""
        return self.y.ndim > 1

    @property
    def score_r2(self) -> float:
        """The score of the resulting model as R^2."""
        if not self.is_fit:
            raise ValueError("model not fitted! no r2")
        return r2_score(self.y, self.yp)

    @property
    def coef_mat(self) -> pd.DataFrame:
        """The coefficient matrix if possible."""
        if not self.is_fit:
            raise ValueError("model not fitted! no coef_mat")
        if self.multioutput:
            # each element is a 'multioutputregressor' object
            self._coef_mat = _coefficient_multioutput(self.fitted, self.X, self.y, self.cv)
        else:
            self._coef_mat = _get_coefficient_normal(self.fitted, self.X)
        return self._coef_mat

    """ ################################ HIDDEN/OVERRIDES ############################################# """

    def __repr__(self):
        if self.is_fit:
            return "MetaML(X: {}, Y: {}, cv={}, score={})".format(self.X.shape, self.y.shape, self.cv, self.score_r2)
        else:
            return "MetaML(X: {}, Y: {}, cv={})".format(self.X.shape, self.y.shape, self.cv)

    """ ################################ FUNCTIONS ############################################# """

    def fit(self):
        """
        Performs a single run/fit and returns predictions and scores based on defaults.
        """
        # perform cross-validated scores, models
        self._grid = GridSearchCV(self.lm,
                                  param_grid={},
                                  scoring="r2",
                                  cv=self.cv,
                                  refit=True,
                                  return_train_score=True,
                                  )
        self._grid.fit(self._x_numpy, self.y)
        # cross-validate the best model
        _scores = cross_validate(self.lm, self._x_numpy, self.y, scoring="r2",
                                 cv=self.cv, return_train_score=True, return_estimator=True)
        # perform cross-validated-predictions
        self.yp = _wrap_pandas(cross_val_predict(self.lm, self._x_numpy, self.y, cv=self.cv), self._y_names)

        # handle cross_validate
        self.fitted = _scores["estimator"]
        self.score_train = np.clip(_scores["train_score"], 0, 1)
        self.score_fit = _scores["fit_time"]
        self.score_test = np.clip(_scores["test_score"], 0, 1)

        self.is_fit = True
        return self
