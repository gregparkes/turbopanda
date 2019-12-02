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
from sklearn.model_selection import cross_val_predict, cross_validate

from .pipes import ml_pipe
from .metrics import correlate

__sklearn_model_packages__ = [tree, linear_model, ensemble, svm, gaussian_process,
                              neighbors]


def _get_hidden_coefficients(directory):
    return [d for d in directory if d.endswith("_") and d.islower()]


def _get_multioutput_wrap(model_t):
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
    return pd.DataFrame(data, columns=labels) if data.ndim > 1 else pd.Series(data, index=labels)


def _get_coefficient_matrix(fitted_models, model_string, X):
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
        self.mdf_ = mp.compute(ml_pipe(mp, x_select, y_select), inplace=False)
        # call cache x and y selectors
        self.mdf_.cache("input", x_select)
        self.mdf_.cache("output", y_select)
        # create X and y
        self.X, self.y = self.mdf_[x_select], self.mdf_[y_select]
        # defaults
        self.cv = 10
        # find and instantiate
        model_t = _find_sklearn_model(model)()
        # cover for multioutput
        if self.y.ndim > 1:
            self.lm = _get_multioutput_wrap(model_t)
        else:
            self.lm = model_t

        self.fit = False

    def single_run(self, scoring="r2"):
        """
        Performs a single run and returns predictions and scores based on defaults.

        Parameters
        -------
        scoring : str
            A scoring method in sklearn, by default use R-squared.
        """
        # preprocess X incase we are just one column
        X_r = self.X.values.reshape(-1, 1) if self.X.ndim == 1 else self.X.values
        # perform cross-validated scores, models
        _scores = cross_validate(self.lm, X_r, self.y,
                                 cv=self.cv, return_estimator=True,
                                 return_train_score=True,
                                 scoring="r2")
        # perform cross-validated-predictions
        self.yp = _wrap_pandas(cross_val_predict(self.lm, X_r, self.y, cv=self.cv), self.X.index)
        # calculate scores
        self.score_pred = correlate(self.y, self.yp, method="r2")

        # handle cross_validate
        self.fitted = _scores["estimator"]
        self.score_train = _scores["train_score"]
        self.score_fit = _scores["fit_time"]
        self.score_test = _scores["test_score"]
        # attempts to get weights
        self.coef_mat = _get_coefficient_matrix(self.fitted, self.model_str, self.X)

        self.fit = True
        return self




    def __repr__(self):
        if self.fit:
            return "MetaML(X: {}, Y: {}, cv={}, score={})".format(self.X.shape, self.y.shape, self.cv, self.score_pred)
        else:
            return "MetaML(X: {}, Y: {}, cv={})".format(self.X.shape, self.y.shape, self.cv)
