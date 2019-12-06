#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:39:37 2019

@author: gparkes
"""
import pandas as pd
from numpy import nan

from sklearn import linear_model as lm
from sklearn import svm, tree
from sklearn import ensemble as ens
from sklearn import gaussian_process as gp
from sklearn import neighbors as nn
from sklearn import neural_network as neu


def param_types():
    return pd.DataFrame({
            'Context': ['lasso, ridge, elasticnet', 'logistic, svm',
               'adaboost, gbt, forest', 'gp', 'decisiontree, gbt, forest', 'nn',
               'mlp', 'svm', 'nn', 'nn', 'forest, decisiontree', 'gbt',
               'gbt, adaboost', 'logistic', 'elasticnet'],
            'DataType': ['float', 'float', 'int', 'object', 'int', 'int', 'int', 'category',
               'category', 'int', 'category', 'category', 'float', 'category',
               'float'],
            "Default": [1.0, 1.0, 50, nan, 3, 4, 10, "rbf", "auto", 3, "entropy", "deviance", 0.1, "l2", 0.5],
            'Range Min': [1.0e-03, 1.0e-02, 1.0e+01, nan, 2.0e+00, 2.0e+00, 1.0e+01,
                nan, nan, 1.5e+01, nan, nan, 1.0e-02, nan, 5.0e-02],
            'Range Max': [1., 5., 500., nan, 5., 8., 40., nan,
                nan, 200., nan, nan, 0.5, nan, 0.95],
            'Options': [nan, nan, nan, nan, nan, nan, nan, 'rbf, poly, sigmoid, auto',
                'auto, ball_tree, kd_tree, brute', nan, 'gini, entropy',
                'deviance, exponential', nan, 'l1, l2, elasticnet', nan],
            'Suggested N': [100., 100.,   8.,  nan,   4.,   4.,   3.,  nan,  nan,   8.,  nan,
                nan,  10.,  nan,  10.],
            'Scale': ['log', 'log', 'normal', nan, 'normal', 'normal', 'normal', nan,
                nan, 'normal', nan, nan, 'log', nan, 'normal']
            }, index=['alpha', 'C', 'n_estimators', 'kernel', 'max_depth', 'n_neighbors',
               'hidden_layer_sizes', 'gamma', 'algorithm', 'leaf_size',
               'criterion', 'loss', 'learning_rate', 'penalty', 'l1_ratio'])


def model_types():
    return pd.DataFrame({
        'Name': ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net',
           'Logistic Regression', 'Gaussian Process', 'Gaussian Process',
           'Ada Boosting', 'Ada Boosting', 'Gradient-Boosted Trees',
           'Gradient-Boosted Trees', 'Decision Tree', 'Decision Tree',
           'Random Forest', 'Random Forest', 'Nearest Neighbors',
           'Nearest Neighbors', 'Multi-layer Perceptron',
           'Multi-layer Perceptron', 'Support Vector Machine',
           'Support Vector Machine'],
        'Short': ['lm', 'lasso', 'ridge', 'elasticnet', 'logistic', 'gp', 'gp',
           'adaboost', 'adaboost', 'gbt', 'gbt', 'decisiontree',
           'decisiontree', 'forest', 'forest', 'nn', 'nn', 'mlp', 'mlp',
           'svm', 'svm'],
        'Model Type': ['regression', 'regression', 'regression', 'regression',
           'classification', 'classification', 'regression', 'classification',
           'regression', 'classification', 'regression', 'classification',
           'regression', 'classification', 'regression', 'classification',
           'regression', 'classification', 'regression', 'classification',
           'regression'],
        'Primary Parameter': [nan, 'alpha', 'alpha', 'alpha', 'C', 'kernel', 'kernel',
           'n_estimators', 'n_estimators', 'n_estimators', 'n_estimators',
           'max_depth', 'max_depth', 'n_estimators', 'n_estimators',
           'n_neighbors', 'n_neighbors', 'hidden_layer_sizes',
           'hidden_layer_sizes', 'C', 'C'],
        'All Parameters': [nan, 'alpha', 'alpha', 'alpha, l1_ratio', 'C, penalty', 'kernel',
           'kernel', 'n_estimators, learning_rate',
           'n_estimators, learning_rate',
           'n_estimators, loss, learning_rate, subsample, max_depth',
           'n_estimators, loss, learning_rate, subsample, max_depth',
           'max_depth, criterion', 'max_depth, criterion',
           'n_estimators, max_depth, criterion, max_features',
           'n_estimators, max_depth, criterion, max_features',
           'n_neighbors, algorithm, leaf_size',
           'n_neighbors, algorithm, leaf_size',
           'hidden_layer_sizes, activation, solver, alpha, learning_rate',
           'hidden_layer_sizes, activation, solver, alpha, learning_rate',
           'C, kernel, gamma', 'C, kernel, gamma'],
        'Object': [
            lm.LinearRegression, lm.Lasso, lm.Ridge, lm.ElasticNet,
            lm.LogisticRegression, gp.GaussianProcessClassifier,
            gp.GaussianProcessRegressor, ens.AdaBoostClassifier, ens.AdaBoostRegressor,
            ens.GradientBoostingClassifier, ens.GradientBoostingRegressor,
            tree.DecisionTreeClassifier, tree.DecisionTreeRegressor,
            ens.RandomForestClassifier, ens.RandomForestRegressor,
            nn.KNeighborsClassifier, nn.KNeighborsRegressor,
            neu.MLPClassifier, neu.MLPRegressor, svm.SVC, svm.SVR],
        }, index=['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet',
           'LogisticRegression', 'GaussianProcessClassifier',
           'GaussianProcessRegressor', 'AdaBoostClassifier',
           'AdaBoostRegressor', 'GradientBoostingClassifier ',
           'GradientBoostingRegressor', 'DecisionTreeClassifier',
           'DecisionTreeRegressor', 'RandomForestClassifier',
           'RandomForestRegressor', 'KNeighborsClassifier',
           'KNeighborsRegressor', 'MLPClassifier', 'MLPRegressor', 'SVC',
           'SVR'])