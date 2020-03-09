#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit default parameters for varying scikit-learn models."""

import numpy as np
from numpy import nan
from pandas import DataFrame

__all__ = ('param_types', 'model_types')


def param_types():
    """Returns all of the supported sklearn parameter types.

    Format: Context, DataType, Default, Range Min, Range Max, Options, Suggested N, Scale

        if datatype is a float or int AND scale is 'log', use logspace[range min, range max, suggested n] as scope
        if datatype is a float or int AND scale is 'normal', use linspace[range min, range max, suggested n] as scope
        if datatype is category: use default, options

    Context informs as to which models the parameter is appropriate with.
    """
    return DataFrame({'alpha': {'Context': 'lasso, ridge, elasticnet',
                                'DataType': 'float',
                                'Default': 1.0,
                                'Range Min': 0.001,
                                'Range Max': 5.0,
                                'Options': nan,
                                'Suggested N': 50.0,
                                'Scale': 'log'},
                      'C': {'Context': 'logistic, svm',
                            'DataType': 'float',
                            'Default': 1.0,
                            'Range Min': 0.01,
                            'Range Max': 5.0,
                            'Options': nan,
                            'Suggested N': 50.0,
                            'Scale': 'log'},
                      'n_estimators': {'Context': 'adaboost, gbt, forest, bagging',
                                       'DataType': 'int',
                                       'Default': 50,
                                       'Range Min': 10.0,
                                       'Range Max': 300.0,
                                       'Options': nan,
                                       'Suggested N': 8.0,
                                       'Scale': 'normal'},
                      'kernel': {'Context': 'gp',
                                 'DataType': 'object',
                                 'Default': nan,
                                 'Range Min': nan,
                                 'Range Max': nan,
                                 'Options': nan,
                                 'Suggested N': nan,
                                 'Scale': nan},
                      'max_depth': {'Context': 'decisiontree, gbt, forest',
                                    'DataType': 'int',
                                    'Default': 3,
                                    'Range Min': 2.0,
                                    'Range Max': 5.0,
                                    'Options': nan,
                                    'Suggested N': 4.0,
                                    'Scale': 'normal'},
                      'n_neighbors': {'Context': 'nn',
                                      'DataType': 'int',
                                      'Default': 4,
                                      'Range Min': 2.0,
                                      'Range Max': 8.0,
                                      'Options': nan,
                                      'Suggested N': 4.0,
                                      'Scale': 'normal'},
                      'hidden_layer_sizes': {'Context': 'mlp',
                                             'DataType': 'int',
                                             'Default': 10,
                                             'Range Min': 10.0,
                                             'Range Max': 40.0,
                                             'Options': nan,
                                             'Suggested N': 3.0,
                                             'Scale': 'normal'},
                      'gamma': {'Context': 'svm',
                                'DataType': 'category',
                                'Default': 'rbf',
                                'Range Min': nan,
                                'Range Max': nan,
                                'Options': 'rbf, poly, sigmoid, auto',
                                'Suggested N': nan,
                                'Scale': nan},
                      'algorithm': {'Context': 'nn',
                                    'DataType': 'category',
                                    'Default': 'auto',
                                    'Range Min': nan,
                                    'Range Max': nan,
                                    'Options': 'auto, ball_tree, kd_tree, brute',
                                    'Suggested N': nan,
                                    'Scale': nan},
                      'leaf_size': {'Context': 'nn',
                                    'DataType': 'int',
                                    'Default': 3,
                                    'Range Min': 15.0,
                                    'Range Max': 200.0,
                                    'Options': nan,
                                    'Suggested N': 8.0,
                                    'Scale': 'normal'},
                      'criterion': {'Context': 'forest',
                                    'DataType': 'category',
                                    'Default': 'entropy',
                                    'Range Min': nan,
                                    'Range Max': nan,
                                    'Options': 'gini, entropy',
                                    'Suggested N': nan,
                                    'Scale': nan},
                      'loss': {'Context': 'gbt',
                               'DataType': 'category',
                               'Default': 'deviance',
                               'Range Min': nan,
                               'Range Max': nan,
                               'Options': 'deviance, exponential',
                               'Suggested N': nan,
                               'Scale': nan},
                      'learning_rate': {'Context': 'gbt, adaboost',
                                        'DataType': 'float',
                                        'Default': 0.1,
                                        'Range Min': 0.01,
                                        'Range Max': 0.5,
                                        'Options': nan,
                                        'Suggested N': 10.0,
                                        'Scale': 'log'},
                      'penalty': {'Context': 'logistic',
                                  'DataType': 'category',
                                  'Default': 'l2',
                                  'Range Min': nan,
                                  'Range Max': nan,
                                  'Options': 'l1, l2, elasticnet',
                                  'Suggested N': nan,
                                  'Scale': nan},
                      'l1_ratio': {'Context': 'elasticnet',
                                   'DataType': 'float',
                                   'Default': 0.5,
                                   'Range Min': 0.2,
                                   'Range Max': 0.8,
                                   'Options': nan,
                                   'Suggested N': 3,
                                   'Scale': 'normal'},
                      'n_restarts_optimizer': {'Context': 'gauss',
                                               'DataType': 'int',
                                               'Default': 5,
                                               'Range Min': 2,
                                               'Range Max': 15,
                                               'Options': nan,
                                               'Suggested N': 5,
                                               'Scale': 'normal'},
                      'min_samples_split': {'Context': 'decisiontree, forest',
                                            'DataType': 'int',
                                            'Default': 2,
                                            'Range Min': 1,
                                            'Range Max': 7,
                                            'Options': nan,
                                            'Suggested N': 4,
                                            'Scale': 'normal'},
                      'min_samples_leaf': {'Context': 'decisiontree, forest, gbt',
                                           'DataType': 'int',
                                           'Default': 1,
                                           'Range Min': 1,
                                           'Range Max': 5,
                                           'Options': nan,
                                           'Suggested N': 3,
                                           'Scale': 'normal'},
                      'max_features': {'Context': 'decisiontree, forest, gbt, bagging',
                                       'DataType': 'category',
                                       'Default': "auto",
                                       'Range Min': nan,
                                       'Range Max': nan,
                                       'Options': "auto, sqrt, log2",
                                       'Suggested N': nan,
                                       'Scale': nan},
                      }).T


def model_types():
    """Returns all of the supported sklearn model types.

    Format: Name, Short, ModelType, Primary Parameter, All Parameters

    Maps to `param_types`.
    """
    return DataFrame({'LinearRegression': {'Name': 'Linear Regression',
                                           'Short': 'lm',
                                           'ModelType': 'regression',
                                           'Primary Parameter': nan,
                                           'All Parameters': nan},
                      'Lasso': {'Name': 'Lasso',
                                'Short': 'lasso',
                                'ModelType': 'regression',
                                'Primary Parameter': 'alpha',
                                'All Parameters': 'alpha'},
                      'Ridge': {'Name': 'Ridge',
                                'Short': 'ridge',
                                'ModelType': 'regression',
                                'Primary Parameter': 'alpha',
                                'All Parameters': 'alpha'},
                      'ElasticNet': {'Name': 'Elastic Net',
                                     'Short': 'elasticnet',
                                     'ModelType': 'regression',
                                     'Primary Parameter': 'alpha',
                                     'All Parameters': 'alpha, l1_ratio'},
                      'LogisticRegression': {'Name': 'Logistic Regression',
                                             'Short': 'logistic',
                                             'ModelType': 'classification',
                                             'Primary Parameter': 'C',
                                             'All Parameters': 'C, penalty'},
                      'GaussianProcessClassifier': {'Name': 'Gaussian Process',
                                                    'Short': 'gauss',
                                                    'ModelType': 'classification',
                                                    'Primary Parameter': 'kernel',
                                                    'All Parameters': 'n_restarts_optimizer, kernel, max_iter_predict'},
                      'GaussianProcessRegressor': {'Name': 'Gaussian Process',
                                                   'Short': 'gauss',
                                                   'ModelType': 'regression',
                                                   'Primary Parameter': 'kernel',
                                                   'All Parameters': 'n_restarts_optimizer, kernel, max_iter_predict'},
                      'PassiveAggressiveClassifier': {'Name': 'Robust Regression',
                                                      'Short': 'robust',
                                                      'ModelType': 'classification',
                                                      'Primary Parameter': 'C',
                                                      'All Parameters': 'C, max_iter, tol'},
                      'PassiveAggressiveRegressor': {'Name': 'Robust Regression',
                                                     'Short': 'robust',
                                                     'ModelType': 'regression',
                                                     'Primary Parameter': 'C',
                                                     'All Parameters': 'C, max_iter, tol'},
                      'AdaBoostClassifier': {'Name': 'Ada Boosting',
                                             'Short': 'adaboost',
                                             'ModelType': 'classification',
                                             'Primary Parameter': 'n_estimators',
                                             'All Parameters': 'n_estimators, learning_rate'},
                      'AdaBoostRegressor': {'Name': 'Ada Boosting',
                                            'Short': 'adaboost',
                                            'ModelType': 'regression',
                                            'Primary Parameter': 'n_estimators',
                                            'All Parameters': 'n_estimators, learning_rate'},
                      'GradientBoostingClassifier ': {'Name': 'Gradient-Boosted Trees',
                                                      'Short': 'gbt',
                                                      'ModelType': 'classification',
                                                      'Primary Parameter': 'n_estimators',
                                                      'All Parameters': 'n_estimators, loss, learning_rate, subsample, max_depth'},
                      'GradientBoostingRegressor': {'Name': 'Gradient-Boosted Trees',
                                                    'Short': 'gbt',
                                                    'ModelType': 'regression',
                                                    'Primary Parameter': 'n_estimators',
                                                    'All Parameters': 'n_estimators, loss, learning_rate, subsample, max_depth'},
                      'DecisionTreeClassifier': {'Name': 'Decision Tree',
                                                 'Short': 'decisiontree',
                                                 'ModelType': 'classification',
                                                 'Primary Parameter': 'max_depth',
                                                 'All Parameters': 'max_depth, criterion'},
                      'DecisionTreeRegressor': {'Name': 'Decision Tree',
                                                'Short': 'decisiontree',
                                                'ModelType': 'regression',
                                                'Primary Parameter': 'max_depth',
                                                'All Parameters': 'max_depth, criterion'},
                      'RandomForestClassifier': {'Name': 'Random Forest',
                                                 'Short': 'forest',
                                                 'ModelType': 'classification',
                                                 'Primary Parameter': 'n_estimators',
                                                 'All Parameters': 'n_estimators, max_depth, criterion, max_features'},
                      'RandomForestRegressor': {'Name': 'Random Forest',
                                                'Short': 'forest',
                                                'ModelType': 'regression',
                                                'Primary Parameter': 'n_estimators',
                                                'All Parameters': 'n_estimators, max_depth, criterion, max_features'},
                      'BaggingClassifier': {'Name': 'Bagging Estimator',
                                            'Short': 'bagging',
                                            'ModelType': 'classification',
                                            'Primary Parameter': 'n_estimators',
                                            'All Parameters': 'n_estimators, max_depth, max_features, max_samples'},
                      'BaggingRegressor': {'Name': 'Bagging Estimator',
                                           'Short': 'bagging',
                                           'ModelType': 'regression',
                                           'Primary Parameter': 'n_estimators',
                                           'All Parameters': 'n_estimators, max_depth, max_features, max_samples'},
                      'ExtraTreesClassifier': {'Name': 'Extra Trees',
                                               'Short': 'etc',
                                               'ModelType': 'classification',
                                               'Primary Parameter': 'n_estimators',
                                               'All Parameters': 'n_estimators, max_depth, criterion, '
                                                                 'min_samples_leaf, max_features '
                                               },
                      'ExtraTreesRegressor': {'Name': 'Extra Trees',
                                              'Short': 'etr',
                                              'ModelType': 'regression',
                                              'Primary Parameter': 'n_estimators',
                                              'All Parameters': 'n_estimators, max_depth, criterion, '
                                                                'min_samples_leaf, max_features '

                                              },
                      'KNeighborsClassifier': {'Name': 'Nearest Neighbors',
                                               'Short': 'nn',
                                               'ModelType': 'classification',
                                               'Primary Parameter': 'n_neighbors',
                                               'All Parameters': 'n_neighbors, weights, algorithm, leaf_size'},
                      'KNeighborsRegressor': {'Name': 'Nearest Neighbors',
                                              'Short': 'nn',
                                              'ModelType': 'regression',
                                              'Primary Parameter': 'n_neighbors',
                                              'All Parameters': 'n_neighbors, weights, algorithm, leaf_size'},
                      'MLPClassifier': {'Name': 'Multi-layer Perceptron',
                                        'Short': 'mlp',
                                        'ModelType': 'classification',
                                        'Primary Parameter': 'hidden_layer_sizes',
                                        'All Parameters': 'hidden_layer_sizes, activation, solver, alpha, learning_rate'},
                      'MLPRegressor': {'Name': 'Multi-layer Perceptron',
                                       'Short': 'mlp',
                                       'ModelType': 'regression',
                                       'Primary Parameter': 'hidden_layer_sizes',
                                       'All Parameters': 'hidden_layer_sizes, activation, solver, alpha, learning_rate'},
                      'SVC': {'Name': 'Support Vector Machine',
                              'Short': 'svm',
                              'ModelType': 'classification',
                              'Primary Parameter': 'C',
                              'All Parameters': 'C, kernel, gamma'},
                      'SVR': {'Name': 'Support Vector Machine',
                              'Short': 'svm',
                              'ModelType': 'regression',
                              'Primary Parameter': 'C',
                              'All Parameters': 'C, kernel, gamma'}
                      }).T