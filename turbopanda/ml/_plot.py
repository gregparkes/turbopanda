#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for machine-learning results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from turbopanda.plot import shape_multiplot, color_qualitative, legend
from turbopanda.utils import intersect, difference, listify, set_like, switcheroo, dictzip
from turbopanda._pipe import Pipe
from ._default import model_types, param_types

from ._package import find_model_family

__all__ = ('parameter_tune_plot', 'best_model_plot')


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
            plot.boxplot(-subset['split[0-9]+_test_score'], notch=True)
        else:
            plot.boxplot(subset['split[0-9]+_test_score'], notch=True)
        plot.set_ylabel(score)
        return
    else:
        # model should be unique
        prim_param = _mt.loc[model_name, "Primary Parameter"]
        if prim_param is np.nan:
            return
        # if the score is a negative-scoring method (-mse, -rmse), convert to positive.
        subset_tr = subset.copy()
        if subset_tr['mean_test_score'].mean() < 0.:
            subset_tr.transform(np.abs, "(?:split[0-9]+|mean)_(?:train|test)_score")
        if plot is None:
            fig, plot = plt.subplots(figsize=(6, 4))
        if prim_param in log_params:
            plot.set_xscale("log")
        # check that prim_param is in params
        if len(intersect([prefix+prim_param], params)) <= 0:
            raise ValueError("primary param: '{}' not found in parameter list: {}".format(prim_param, params))

        if len(params) == 1:
            # mean, sd
            test_m = subset_tr['mean_test_score']
            test_sd = subset_tr['std_test_score']
            _p = subset_tr[params[0]]
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
                _p = subset_tr.df_.loc[subset_tr[non_prim] == line, prefix + prim_param]
                test_m = subset_tr.df_.loc[subset_tr[non_prim] == line, 'mean_test_score']
                test_sd = subset_tr.df_.loc[subset_tr[non_prim] == line, 'std_test_score']
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


def best_model_plot(cv_results, minimize=True, score="RMSE", **box_kws):
    """Determines the best model (min or max) and plots the boxplot of all resulting best models.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `fit_grid`.
    minimize : bool
        If True, selects best smallest score, else select best largest score
    score : str
        The name of the scoring function
    box_kws : dict, optional
        Keyword arguments to pass to `plt.boxplot`.

    Returns
    -------
    None
    """
    if "model" in cv_results.columns:
        # create figures
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        # create a copy
        res = cv_results.copy()
        # transform.
        if res['mean_test_score'].mean() < 0.:
            res.transform(np.abs, "(?:split[0-9]+|mean)_(?:train|test)_score")
        # for each 'model', arrange data into boxplot
        if minimize:
            indices = res.df_.groupby("model")['mean_test_score'].idxmin()
        else:
            indices = res.df_.groupby("model")['mean_test_score'].idxmax()
        # arrange data
        result_p = res.df_.loc[indices, res.view("split[0-9]+_test_score")]
        # reorder based on the best score
        re_order = result_p.mean(axis=1).sort_values()
        result_p = result_p.reindex(re_order.index)
        # get best score name
        indices = switcheroo(indices).reindex(re_order.index)
        # plot
        bp = ax.boxplot(result_p, patch_artist=True, **box_kws)
        # fetch package names and map them to colors - returned as pd.Series
        packages = find_model_family(indices.values)
        # map colors to each of the packages.
        mapping = dictzip(
            set_like(packages),
            color_qualitative(len(set_like(packages)))
        )
        mapped_cols = packages.map(mapping)
        # iterate over boxes and colour
        for box, col in zip(bp['boxes'], mapped_cols):
            box.set(facecolor=col, linewidth=1.2)
        plt.setp(bp['medians'], linewidth=1.5)
        # additional box requirements
        ax.set_xlabel("Model")
        ax.set_ylabel(score)
        ax.set_xticklabels(indices.values)
        ax.tick_params("x", rotation=45)
        ax.grid()
        for tick in ax.get_xmajorticklabels():
            tick.set_horizontalalignment('right')
        # generate legend
        ax.legend(legend(mapping), list(mapping.keys()), bbox_to_anchor=(1.03, 1.03))
        plt.show()
    else:
        raise ValueError("column 'model' not found in `cv_results`.")
