#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:12:08 2019

@author: gparkes
"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, PassiveAggressiveRegressor
from sklearn.svm import SVR

def to_pipe(m):
    return Pipeline([("model", m)])


preset_randomforest = {
    "name": "Random Forest",
    "type_y": "regression",
    "param_name": "n_estimators",
    "pipe": to_pipe(RandomForestRegressor()),
    "params": {"model__n_estimators": [20, 50, 100, 150, 200]}
}

preset_lasso = {
    "name": "Lasso",
    "type_y": "regression",
    "param_name": "alpha",
    "pipe": to_pipe(Lasso()),
    "params": {"model__alpha": np.logspace(-2.5, 0., 100)}
}

preset_svm = {
    "name": "Support Vector Machine",
    "type_y": "regression",
    "param_name": "C",
    "pipe": to_pipe(SVR(tol=1e-4, gamma="auto")),
    "params": {"model__C": np.logspace(-2., 2., 150)}
}

preset_ridge = {
    "name": "Ridge",
    "type_y": "regression",
    "param_name": "alpha",
    "pipe": to_pipe(Ridge()),
    "params": {"model__alpha": np.logspace(-2.5, 1.5, 100)}
}

preset_adaboost = {
    "name": "AdaBoost",
    "type_y": "regression",
    "param_name": "n_estimators",
    "pipe": to_pipe(AdaBoostRegressor()),
    "params": {"model__n_estimators": [20, 50, 75, 100, 125, 150, 200]}
}

preset_passiveaggressive = {
    "name": "Robust Regression",
    "type_y": "regression",
    "param_name": "C",
    "pipe": to_pipe(PassiveAggressiveRegressor()),
    "params": {"model__C": np.logspace(-3., 1., 100)}
}