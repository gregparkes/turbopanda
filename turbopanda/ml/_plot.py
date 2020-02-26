#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for machine-learning results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

from turbopanda.corr import correlate
from turbopanda.corr._correlate import _row_to_matrix
from turbopanda.plot import shape_multiplot, color_qualitative
from turbopanda.utils import union, intersect, difference, listify, set_like
from turbopanda._pipe import Pipe

from ._clean import cleaned_subset
from ._default import model_types, param_types
from turbopanda.stats import vif, cook_distance

__all__ = ('coefficient_plot', 'overview_plot', 'parameter_tune_plot', 'best_model_plot')


def _fitted_vs_residual(plot, y, yp):
    plot.scatter(yp, y - yp, color='g', alpha=.5)
    plot.hlines(0, xmin=yp.min(), xmax=yp.max(), color='r')
    plot.set_xlabel("Fitted Values")
    plot.set_ylabel("Residuals")


def _boxplot_scores(plot, cv):
    plot.boxplot(cv['train_score|test_score'].values)
    plot.set_xlabel("Train/Test Score")
    plot.set_ylabel(r"$r^2$")
    plot.tick_params('x', rotation=45)
    plot.set_xticks(range(1, 3))
    plot.set_xticklabels(['test', 'train'])


def _actual_vs_predicted(plot, y, yp):
    # KDE plot estimation between y and yhat
    plot.scatter(y, yp, color='b', alpha=.5)
    plot.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], 'k-')
    plot.set_xlabel("Actual Values")
    plot.set_ylabel("Fitted Values")


def coefficient_plot(plot, cv):
    """Plots the coefficients from a cv results."""
    coef = cv['w__']
    # sort the coefficients by size
    coef = coef.reindex(cv['w__'].mean(axis=0).sort_values().index, axis=1)
    # plot
    plot.boxplot(coef.values)
    plot.set_xlabel("Coefficient")

    # if we have too many labels, randomly select some
    if len(coef.columns) > 10:
        # subset
        xtick_locs = np.random.choice(len(coef.columns), 10, replace=False)
        plot.set_xticks(xtick_locs)
        plot.set_xticklabels(coef.iloc[:, xtick_locs].columns.str[3:])
    else:
        plot.set_xticks(range(1, len(coef.columns) + 1))
        plot.set_xticklabels(coef.columns.str[3:])

    plot.tick_params("x", rotation=45)


def _basic_correlation_matrix(plot, df, x, y):
    # determine columns
    cols = union(df.view(x), y)
    # perform correlation
    corr = correlate(df[cols].dropna())
    # convert to matrix
    _cmatrix = _row_to_matrix(corr)
    # plot heatmap
    plot.pcolormesh(_cmatrix, vmin=-1., vmax=1., cmap="seismic")
    # if we have too many labels, randomly choose some
    if _cmatrix.shape[0] > 10:
        tick_locs = np.random.choice(_cmatrix.shape[0], 10, replace=False)
        plot.set_xticks(tick_locs)
        plot.set_xticklabels(_cmatrix.iloc[:, tick_locs].columns)
        plot.set_yticks(tick_locs)
        plot.set_yticklabels(_cmatrix.iloc[:, tick_locs].columns)
        plot.tick_params('x', rotation=45)


def _plot_vif(plot, _vif):
    if isinstance(_vif, pd.Series):
        plot.bar(range(1, len(_vif) + 1), _vif.values, width=.7, color='r')
        plot.set_xlabel("Feature")
        plot.set_ylabel("VIF")
        if len(_vif) > 10:
            tick_locs = np.random.choice(len(_vif), 10, replace=False)
            plot.set_xticks(tick_locs)
            plot.set_xticklabels(_vif.iloc[tick_locs].index.values)
        else:
            plot.set_xticks(range(1, len(_vif) + 1))
            plot.set_xticklabels(_vif.index.values)

        plot.tick_params('x', rotation=45)


def _cooks(plot, df, x, y, yp):
    cooks = cook_distance(df, x, y, yp)
    plot.plot(cooks, 'ko', alpha=.6)
    plot.hlines(0, xmin=0, xmax=cooks.shape[0])
    plot.set_ylabel("Cook's distance")


def overview_plot(df, x, y, cv, yp):
    """Presents an overview of the results of a machine learning basic run.

    Parameters
    ----------
    df : MetaPanda
    x : selector
    y : str
    cv : MetaPanda
    yp : pd.Series

    Returns
    -------
    a bunch of plots. No Return.
    """
    fig, ax = shape_multiplot(8, ax_size=3)
    # set yp as series
    yp = yp[y].squeeze()
    # pair them and remove NA
    _df = cleaned_subset(df, x, y)

    # plot 1. fitted vs. residual plots
    _fitted_vs_residual(ax[0], _df[y], yp)
    # plot 2. boxplot for scores
    _boxplot_scores(ax[1], cv)
    # plot 3. KDE plot estimation between y and yhat
    _actual_vs_predicted(ax[2], _df[y], yp)
    # plot 4. coefficient plot
    coefficient_plot(ax[3], cv)
    # plot 5. correlation matrix
    _basic_correlation_matrix(ax[4], df, x, y)
    # plot 6. variance inflation factor for each explanatory variable
    _plot_vif(ax[5], vif(df, x, y))
    # plot 7. q-q plot
    stats.probplot(_df[y], dist="norm", plot=ax[6])
    # plot 8. outlier detection using cook's distance plot
    """Cook's distance is defined as the sum of all the changes in the regression model when #
    observation i is removed from it."""
    _cooks(ax[7], df, x, y, yp)

    fig.tight_layout()
    plt.show()


def _model_selection_parameters(cv_results, model_name, params, plot=None, prefix="param_model__", score="RMSE"):
    # definition of accepted models.
    _mt = model_types()
    _pt = param_types()
    log_params = ["alpha", "C"]
    # get primary parameter based on the model.
    plot.set_title(model_name)
    # if we only have one row, just compute the boxplot
    subset = cv_results.compute(model_name)

    if subset.n_ == 1:
        if subset['mean_test_score'] < 0.:
            plot.boxplot(-subset['split[0-9]+_test_score'])
        else:
            plot.boxplot(subset['split[0-9]+_test_score'])
        plot.set_xticklabels([subset['model']])
        plot.set_ylabel(score)
        return
    else:
        # model should be unique
        prim_param = _mt.loc[model_name, "Primary Parameter"]
        if prim_param is np.nan:
            return
        # if the score is a negative-scoring method (-mse, -rmse), convert to positive.
        if subset['mean_test_score'].mean() < 0.:
            subset.transform(np.abs, "mean_test_score")
        if plot is None:
            fig, plot = plt.subplots(figsize=(6, 4))
        if prim_param in log_params:
            plot.set_xscale("log")
        # check that prim_param is in params
        if len(intersect([prefix+prim_param], params)) <= 0:
            raise ValueError("primary param: '{}' not found in parameter list: {}".format(prim_param, params))

        if len(params) == 1:
            # mean, sd
            test_m = subset['mean_test_score']
            test_sd = subset['std_test_score']
            _p = subset[params[0]]
            # generate a random qualitative color
            color = color_qualitative(1)[0]
            # plotting
            plot.plot(_p, test_m, 'x-', color=color)
            plot.fill_between(_p, test_m + test_sd, test_m - test_sd, alpha=.3, color=color)
        elif len(params) == 2:
            non_prim = difference([prefix+prim_param], params)[0]
            non_prim_uniq = subset[non_prim].unique()
            colors = color_qualitative(len(non_prim_uniq))
            # fetch non-primary column.
            for line, c in zip(non_prim_uniq, colors):
                # mean, sd
                _p = subset.df_.loc[subset[non_prim] == line, prefix + prim_param]
                test_m = subset.df_.loc[subset[non_prim] == line, 'mean_test_score']
                test_sd = subset.df_.loc[subset[non_prim] == line, 'std_test_score']
                plot.plot(_p, test_m, 'x-', label="{}={}".format(non_prim.split("__")[-1], line), color=c)
                plot.fill_between(_p, test_m + test_sd, test_m - test_sd, alpha=.3, color=c)
            plot.legend(loc="best")

        plot.set_xlabel(prim_param)
        plot.set_ylabel(score)


def parameter_tune_plot(cv_results):
    """Iterates over every model type in `cv_results` and plots the best parameter. cv_results is MetaPanda

    Generates a series of plots for each model type, plotting the parameters.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `fit_grid`.

    Returns
    -------
    None
    """
    # determine the models found within.
    if "model" in cv_results.columns:
        # get unique models
        models = set_like(cv_results['model'])
        # create figures
        fig, axes = shape_multiplot(len(models), ax_size=5)
        for i, m in enumerate(models):
            # add pipelines into cv_results for selection.
            cv_results.cache_pipe(m, Pipe(['filter_rows', lambda df: df['model'] == m]))
            # determine parameter names from results.
            _P = [p for p in cv_results.view("param_model__") if
                  cv_results.df_.loc[cv_results['model'] == m, p].dropna().shape[0] > 0]
            _model_selection_parameters(cv_results, m, _P, axes[i])
        fig.tight_layout()
        plt.show()
    else:
        raise ValueError("column 'model' not found in `cv_results`.")


def best_model_plot(cv_results, minimize=True, score="RMSE"):
    """Determines the best model (min or max) and plots the boxplot of all resulting best models.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `fit_grid`.
    minimize : bool
        If True, selects best smallest score, else select best largest score
    score : str
        The name of the scoring function

    Returns
    -------
    None
    """
    if "model" in cv_results.columns:
        # get unique models
        models = set_like(cv_results['model'])
        # create figures
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        # for each 'model', arrange data into boxplot
        if minimize:
            indices = cv_results.df_.groupby("model")['mean_test_score'].idxmin()
        else:
            indices = cv_results.df_.groupby("model")['mean_test_score'].idxmax()
        # transform.
        if cv_results['mean_test_score'].mean() < 0.:
            cv_results.transform(np.abs, "mean_test_score")
        # arrange data
        result_p = cv_results.df_.loc[indices, cv_results.view("split[0-9]+_test_score")]
        # plot
        ax.boxplot(result_p)
        ax.set_xlabel("Model")
        ax.set_ylabel(score)
        ax.set_xticklabels(models.values)
        plt.show()
    else:
        raise ValueError("column 'model' not found in `cv_results`.")
