#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides plot_learning for data returned from `fit_learning`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from turbopanda.plot import color_qualitative, gridplot


def learning_curve_plot(cv_results,
                        perm=None,
                        plot_size=3,
                        score="RMSE",
                        inverse_score=True):
    """Plot a learning curve of the parameters from a call to `fit_learning`.

    Parameters
    ----------
    cv_results : MetaPanda
        The results from `fit_learning`
    perm : np.ndarray, optional
        Permutation results if conducted
    plot_size : int, optional
        The plot sizes of the graphs
    score : str
        The name of the scoring function, default RMSE
    inverse_score : bool
        If True, uses absolute on train/test scores

    Returns
    -------
    None
    """
    fig, axes = gridplot(3, arrange='column', ax_size=plot_size)

    _cv = cv_results.copy()
    if inverse_score:
        _cv.transform(np.abs, '(mean|std)_(train|test)_score', whole=True)

    # 1. plotting the learning curve
    axes[0].set_title("Learning Curves")
    axes[0].plot(_cv['N'], _cv['mean_test_score'], 'x-', color='r', label='test')
    axes[0].fill_between(
        _cv['N'],
        _cv['mean_test_score'] + _cv['std_test_score'],
        _cv['mean_test_score'] - _cv['std_test_score'], color='r', alpha=.3
    )
    axes[0].plot(_cv['N'], _cv['mean_train_score'], 'x-', color='g', label='train')
    axes[0].fill_between(
        _cv['N'],
        _cv['mean_train_score'] + _cv['std_train_score'],
        _cv['mean_train_score'] - _cv['std_train_score'], color='g', alpha=.3
    )
    axes[0].legend(loc='best')
    axes[0].set_xlabel(r"$N$ train sizes")
    axes[0].set_ylabel(score)

    # 2. model scalability
    axes[1].set_title("Model Scalability")
    axes[1].plot(_cv['N'], _cv['mean_fit_time'], 'x-', color='b')
    axes[1].fill_between(
        _cv['N'],
        _cv['mean_fit_time'] + _cv['std_fit_time'],
        _cv['mean_fit_time'] - _cv['std_fit_time'], color='b', alpha=.3
    )
    axes[1].set_xlabel(r"$N$ train sizes")
    axes[1].set_ylabel("Fit Times")

    # 3. Model performance
    axes[2].set_title("Model Performance")
    axes[2].plot(_cv['mean_fit_time'], _cv['mean_test_score'], 'x-', color='orange')
    axes[2].fill_between(
        _cv['mean_fit_time'],
        _cv['mean_test_score'] + _cv['std_test_score'],
        _cv['mean_test_score'] - _cv['std_test_score'], color='orange',
        alpha=.3
    )
    axes[2].set_xlabel(r"Fit Times")
    axes[2].set_ylabel(score)

    fig.tight_layout()
    plt.show()
