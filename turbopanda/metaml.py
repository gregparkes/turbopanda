#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:01:53 2019

@author: gparkes

Creates Meta Scikit-learn models.
"""
import numpy as np
import pandas as pd

from sklearn import tree, linear_model, ensemble, svm, gaussian_process, neighbors
from sklearn.base import is_classifier, is_regressor, BaseEstimator
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import r2_score

from .pipes import ml_regression_pipe
from .models import *

__sklearn_model_packages__ = [tree, linear_model, ensemble, svm, gaussian_process,
                              neighbors]


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


def _get_coefficient_matrix(fitted_models, X):
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


class MetaML(object):
    """
    An object that handles complex operations on a MetaPanda object to run
    scikit-learn models.
    """

    def __init__(self, mp, x_select, y_select, model="LinearRegression"):
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
        model : str, sklearn model
            The model selected to perform a run as
        """
        # make mp ML-ready
        self.model_str = model
        # compute ML pipeline to dataset
        self.mdf_ = mp.compute(ml_regression_pipe(mp, x_select, y_select), inplace=False)
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
        if prim_param != np.nan:
            def_value = _pt.loc[prim_param, "Default"]
            # set parameter of primary value
            model_t.set_params(**{prim_param:def_value})

        # cover for multioutput
        if self.multioutput:
            self.lm = _multioutput_wrap(model_t)
        else:
            self.lm = model_t

        self.is_fit = False

    """ ################################ PROPERTIES #############################################"""

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, _cv):
        if isinstance(_cv, (int, np.int)):
            self._cv = _cv
        else:
            raise TypeError("cv is not of type [int]")

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _x):
        if isinstance(_x, (pd.DataFrame, pd.Series)):
            self._X = _x
        else:
            raise TypeError("X input is not of type [pd.DataFrame, pd.Series]")

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _y):
        if isinstance(_y, (pd.DataFrame, pd.Series)):
            self._y = _y
        else:
            raise TypeError("y output is not of type [pd.DataFrame, pd.Series]")

    @property
    def model_str(self):
        return self._model_str

    @model_str.setter
    def model_str(self, ms):

        self._model_str = _find_sklearn_model(ms)

    @property
    def _x_names(self):
        return self.X.columns if self.X.ndim > 1 else [self.X.name]

    @property
    def _x_numpy(self):
        return self.X.values.reshape(-1, 1) if self.X.ndim == 1 else self.X.values

    @property
    def _y_names(self):
        return self.y.columns if self.multioutput else self.y.name

    @property
    def multioutput(self):
        return self.y.ndim > 1

    @property
    def score_r2(self):
        if not self.is_fit:
            raise ValueError("model not fitted! no r2")
        return r2_score(self.y, self.yp)

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

        Parameters
        -------
        scoring : str
            A scoring method in sklearn, by default use R-squared.
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
        # attempts to get weights
        self.coef_mat = _get_coefficient_matrix(self.fitted, self.X)

        self.is_fit = True
        return self
