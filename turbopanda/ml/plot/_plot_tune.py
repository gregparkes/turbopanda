#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for tuning parameters for ML models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbopanda.pipe import absolute, filter_rows_by_column
from turbopanda.plot import color_qualitative, gridplot, box1d
from turbopanda.utils import belongs, difference, set_like, instance_check, nonnegative
from turbopanda.str import pattern

from turbopanda.ml._default import model_types, param_types


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
    log_params = ("alpha", "C")
    # get primary parameter based on the model.
    plot.set_title(model_name)

    # subset is pandas.DataFrame
    subset = (cv_results
              .pipe(filter_rows_by_column, lambda z: z['model'] == model_name)
              .pipe(absolute, "(?:split[0-9]+|mean)_(?:train|test)_score")
              )

    if plot is None:
        fig, plot = plt.subplots(figsize=(6, 4))

    if subset.shape[0] == 1:
        pcols = pattern('split[0-9]+_%s_score' % y_name, subset, False)
        box1d(np.asarray(subset[pcols]), ax=plot, label=score, notch=True)
        return
    else:
        # define primary parameter axis as x
        prim_param = params[0] if len(params) == 1 else _comp_p(_mt.loc[model_name, "Primary Parameter"], prefix=prefix)
        # determine x-axis scale
        if _basic_p(prim_param) in log_params:
            plot.set_xscale("log")
        # sort values based on x
        subset_sorted = subset.sort_values(by=prim_param)

        # the number of dimensions affects how we plot this.
        if len(params) == 1:
            # mean, sd
            test_m = subset_sorted['mean_%s_score' % y_name]
            test_sd = subset_sorted['std_%s_score' % y_name]
            # coerce p into float if it is.
            _p = pd.to_numeric(subset_sorted[prim_param], errors="ignore")
            # generate a random qualitative color
            color = color_qualitative(1)[0]
            # plotting
            plot.plot(_p, test_m, 'x-', color=color)
            plot.fill_between(_p,
                              test_m + test_sd,
                              test_m - test_sd,
                              alpha=.3,
                              color=color)
        elif len(params) == 2:
            non_prim = difference([prim_param], params)[0]
            non_prim_uniq = subset_sorted[non_prim].unique()
            colors = color_qualitative(len(non_prim_uniq))
            # fetch non-primary column.
            for line, c in zip(non_prim_uniq, colors):
                # mean, sd
                _p = pd.to_numeric(subset_sorted.loc[subset_sorted[non_prim] == line, prim_param],
                                   errors="ignore")
                test_m = subset_sorted.loc[subset_sorted[non_prim] == line,
                                           'mean_%s_score' % y_name]
                test_sd = subset_sorted.loc[subset_sorted[non_prim] == line,
                                            'std_%s_score' % y_name]
                # organise plot label based on prim param type
                lf = "{}={:0.3f}" if np.isreal(line) else "{}={}"
                plot.plot(_p, test_m, 'x-',
                          label=lf.format(non_prim.split("__")[-1], line), color=c)
                plot.fill_between(_p,
                                  test_m + test_sd,
                                  test_m - test_sd,
                                  alpha=.3,
                                  color=c)
            plot.legend(loc="best")

        plot.set_xlabel(_basic_p(prim_param))
        plot.set_ylabel("%s %s" % (y_name, score))


def parameter_tune(cv_results,
                   y: str = 'test',
                   arrange: str = 'square',
                   ax_size: int = 4):
    """Iterates over every model type in `cv_results` and plots the best parameter.

    Generates a series of plots for each model type, plotting the parameters.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `.fit.grid`.
    y : str, default='test'
        Choose from {'test', 'train'}, determines whether to plot training or testing data
        Uses the mean/std test scores, respectively.
    arrange : str, default='square'
        Choose from {'square', 'row' 'column'}. Indicates preference for direction of plots.
    ax_size : int, default=4
        The default size of each plot

    Returns
    -------
    None
    """
    belongs(y, ('test', 'train'))
    belongs(arrange, ('square', 'column', 'row'))
    nonnegative(ax_size, int)

    # make cv_results dataframe
    cv_results = cv_results.df_ if not isinstance(cv_results, pd.DataFrame) else cv_results

    # determine the models found within.
    if "model" in cv_results.columns:
        # get unique models
        models = set_like(cv_results['model'])
        # create figures
        fig, axes = gridplot(len(models), ax_size=ax_size, arrange=arrange)
        for i, m in enumerate(models):
            # determine parameter names from results.
            param_columns = pattern("param_model__", cv_results.columns, False)
            _P = [p for p in param_columns if
                  cv_results.loc[cv_results['model'] == m, p].dropna().shape[0] > 0]
            _model_selection_parameters(cv_results, m, _P, axes[i], y_name=y)
        fig.tight_layout()
        plt.show()
    else:
        raise ValueError("column 'model' not found in `cv_results`.")
