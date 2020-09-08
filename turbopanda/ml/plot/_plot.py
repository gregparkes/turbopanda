#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for machine-learning results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbopanda.plot import color_qualitative, legend_line
from turbopanda.utils import belongs, dictzip, instance_check, set_like, switcheroo
from turbopanda.ml._package import find_model_family
from turbopanda.str import pattern
from turbopanda.pipe import absolute


def best_model(cv_results: "MetaPanda",
               y_var: str = "test",
               minimize: bool = True,
               score: str = "RMSE",
               **box_kws):
    """Determines the best model (min or max) and plots the boxplot of all resulting best models.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from a call to `fit_grid`.
    y_var : str
        Choose from {'test', 'train'}
        If 'test': draws the test score
        If 'train': draws the training score
    minimize : bool
        If True, selects best smallest score, else select best largest score
    score : str
        The name of the scoring function
    box_kws : dict, optional
        Keyword arguments to pass to `plt.boxplot`.

    Returns
    -------
    fig : matplotlib.figure
        The figure object
    """
    instance_check(minimize, bool)
    instance_check(score, str)
    belongs(y_var, ('train', 'test'))

    sely = pattern("mean_%s_score" % y_var, cv_results.columns, False)
    # create figures
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    # create a copy
    res = cv_results.df_ if not isinstance(cv_results, pd.DataFrame) else cv_results
    # transform.
    if res[sely].mean() < 0.:
        res = res.pipe(absolute, "(?:split[0-9]+|mean)_(?:train|test)_score")
    # for each 'model', arrange data into boxplot
    if minimize:
        indices = res.groupby("model")[sely].idxmin()
    else:
        indices = res.groupby("model")[sely].idxmax()
    # arrange data
    result_p = res.df_.loc[indices, res.view("split[0-9]+_%s_score" % y_var)]
    # reorder based on the best score
    re_order = result_p.median(axis=1).sort_values()
    result_p = result_p.reindex(re_order.index)
    # get best score name
    indices = switcheroo(indices).reindex(re_order.index)
    # plot
    bp = ax.boxplot(result_p, patch_artist=True, showfliers=False, **box_kws)
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
    ax.set_ylabel("%s %s" % (y_var, score))
    ax.set_xticklabels(indices.values)
    ax.tick_params("x", rotation=45)
    ax.grid()
    for tick in ax.get_xmajorticklabels():
        tick.set_horizontalalignment('right')
    # generate legend
    ax.legend(legend_line(mapping), list(mapping.keys()), bbox_to_anchor=(1.03, 1.03))
    plt.show()

    return fig
