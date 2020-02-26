#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:44:25 2019

@author: gparkes

Some functions for plotting code in.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.model_selection import learning_curve


def mean_test(cv_results):
    return cv_results[cv_results.columns[cv_results.columns.str.contains("^split[0-9]+_test_score$")]]


def mean_train(cv_results):
    return cv_results[cv_results.columns[cv_results.columns.str.contains("^split[0-9]+_train_score$")]]


def best_param(cv_results, param_name, as_min=True):
    mt = mean_test(cv_results)
    p = pd.to_numeric(cv_results["param_model__{}".format(param_name)])
    if as_min:
        return p[mt.mean(axis=1).idxmin()]
    else:
        return p[mt.mean(axis=1).idxmax()]


def best_error(cv_results, as_min=True):
    mt = mean_test(cv_results)
    if as_min:
        return mt.mean(axis=1).min()
    else:
        return mt.mean(axis=1).max()


def plot_parameter(cv_results, ax=None, param_name="alpha",
                   title="", logx=True, score="mse"):
    mt = mean_test(cv_results)
    p = cv_results["param_model__{}".format(param_name)].astype(np.float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))

    b_alpha = best_param(cv_results, param_name)
    b_error_m = best_error(cv_results)

    if logx:
        ax.set_xscale("log")

    ax.plot(p, mt.mean(axis=1), 'x', label="%s: %.3f" % (score, b_error_m))
    ax.fill_between(p, mt.mean(axis=1) + mt.std(axis=1),
                    mt.mean(axis=1) - mt.std(axis=1), alpha=.3)
    ax.vlines([b_alpha], ymin=mt.mean(axis=1).min(), ymax=mt.mean(axis=1).max(), linestyle="--")
    ax.set_xlabel(param_name)
    ax.set_ylabel(score)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    fig, ax = plt.subplots()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring="neg_mean_squared_error", train_sizes=train_sizes)

    train_scores_mean = np.mean((train_scores), axis=1)
    train_scores_std = np.std((train_scores), axis=1)
    test_scores_mean = np.mean((test_scores), axis=1)
    test_scores_std = np.std((test_scores), axis=1)
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    print("Min cv-score: %.3f" % test_scores_mean.min())
    ax.legend(loc="best")
    return fig
