#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for tuning parameters for ML models."""

import numpy as np
import matplotlib.pyplot as plt

from turbopanda.plot import color_qualitative, gridplot
from turbopanda.utils import difference, set_like, belongs
from ._default import model_types, param_types
from turbopanda._pipe import Pipe


def _basic_p(comp):
    return comp.rsplit("__", 1)[-1]


def _comp_p(basic, prefix="param_model__"):
    return prefix + basic


def _model_selection_parameters(cv_results,
                                model_name,
                                params,
                                plot=None,
                                y_name="test",
                                prefix="param_model__",
                                score="RMSE"):
    # definition of accepted models.
    _mt = model_types()
    _pt = param_types()
    log_params = ("alpha", "C")
    # get primary parameter based on the model.
    plot.set_title(model_name)
    frp = Pipe(['filter_rows', lambda df: df['model'] == model_name],
               ['transform', np.abs, "(?:split[0-9]+|mean)_(?:train|test)_score"])
    # if we only have one row, just compute the boxplot
    subset = cv_results.compute(frp)

    if plot is None:
        fig, plot = plt.subplots(figsize=(6, 4))

    if subset.n_ == 1:
        plot.boxplot(subset['split[0-9]+_%s_score' % y_name], notch=True)
        plot.set_ylabel(score)
        return
    else:
        # define primary parameter axis as x
        prim_param = params[0] if len(params) == 1 else _comp_p(_mt.loc[model_name, "Primary Parameter"], prefix=prefix)
        # determine x-axis scale
        if _basic_p(prim_param) in log_params:
            plot.set_xscale("log")
        # sort values based on x
        fr_sort = Pipe(['apply', 'sort_values', 'by={}'.format(prim_param)])
        subset_sorted = subset.compute(fr_sort)

        # the number of dimensions affects how we plot this.
        if len(params) == 1:
            # mean, sd
            test_m = subset_sorted['mean_%s_score' % y_name]
            test_sd = subset_sorted['std_%s_score' % y_name]
            _p = subset_sorted[prim_param]
            # generate a random qualitative color
            color = color_qualitative(1)[0]
            # plotting
            plot.plot(_p, test_m, 'x-', color=color)
            plot.fill_between(_p, test_m + test_sd, test_m - test_sd, alpha=.3, color=color)
        elif len(params) == 2:
            non_prim = difference([prim_param], params)[0]
            non_prim_uniq = subset_sorted[non_prim].unique()
            colors = color_qualitative(len(non_prim_uniq))
            # fetch non-primary column.
            for line, c in zip(non_prim_uniq, colors):
                # mean, sd
                _p = subset_sorted.df_.loc[subset_sorted[non_prim] == line, prim_param]
                test_m = subset_sorted.df_.loc[subset_sorted[non_prim] == line, 'mean_%s_score' % y_name]
                test_sd = subset_sorted.df_.loc[subset_sorted[non_prim] == line, 'std_%s_score' % y_name]

                plot.plot(_p, test_m, 'x-', label="{}={}".format(non_prim.split("__")[-1], line), color=c)
                plot.fill_between(_p, test_m + test_sd, test_m - test_sd, alpha=.3, color=c)
            plot.legend(loc="best")

        plot.set_xlabel(_basic_p(prim_param))
        plot.set_ylabel("%s %s" % (y_name, score))


def parameter_tune_plot(cv_results, y='test'):
    """Iterates over every model type in `cv_results` and plots the best parameter. cv_results is MetaPanda

    Generates a series of plots for each model type, plotting the parameters.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `fit_grid`.
    y : str, optional
        Choose from {'test', 'train'}
        Uses the mean/std test or train scores, respectively.

    Returns
    -------
    None
    """
    belongs(y, ('test', 'train'))
    # determine the models found within.
    if "model" in cv_results.columns:
        # get unique models
        models = set_like(cv_results['model'])
        # create figures
        fig, axes = gridplot(len(models), ax_size=4)
        for i, m in enumerate(models):
            # determine parameter names from results.
            _P = [p for p in cv_results.view("param_model__") if
                  cv_results.df_.loc[cv_results['model'] == m, p].dropna().shape[0] > 0]
            _model_selection_parameters(cv_results, m, _P, axes[i], y_name=y)
        fig.tight_layout()
        plt.show()
    else:
        raise ValueError("column 'model' not found in `cv_results`.")