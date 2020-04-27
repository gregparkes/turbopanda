#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an overview for the analysis of PCA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbopanda.plot import gridplot, scatter, annotate
from turbopanda.utils import join, instance_check, nonnegative


def _plot_pca_scatter(model, ax, dist_col: bool):
    # calculate the magnitude away from origin (0, 0)
    mag = np.linalg.norm(model.components_[:, :2], axis=1)
    _x, _y = model.components_[:, 0], model.components_[:, 1]
    _xl, _yl = ["PC%d %.2f" % (i + 1, model.explained_variance_ratio_[i] * 100) for i in range(2)]
    # plot
    if dist_col:
        scatter(_x, _y, c=mag, cmap="RdBu_r",
                x_label=_xl, y_label=_yl, colorbar=False,
                ax=ax)
    else:
        scatter(_x, _y, x_label=_xl, y_label=_yl, colorbar=False,
                ax=ax)
    # add pca lines
    ax.hlines(0, xmin=_x.min(), xmax=_x.max(), linestyle="--")
    ax.vlines(0, ymin=_y.min(), ymax=_y.max(), linestyle="--")
    ax.grid()


def _explained_variance_plot(model, ax, cutoff=.9):
    n = len(model.explained_variance_ratio_)
    _x = np.arange(0, n + 1)
    _y = np.hstack((np.array([0]), model.explained_variance_ratio_))
    _ycum = np.cumsum(_y)
    best_index = np.where(_ycum > cutoff)[0][0]
    # calculate AUC
    auc = np.trapz(_ycum, _x / (n+1))
    # plot
    ax.plot(_x, _ycum, "x-")
    # plot best point
    ax.scatter([_x[best_index]], [_ycum[best_index]], facecolors="None",
               edgecolors="red", s=100,
               label="n=%d, auc=%.3f" % (_x[best_index], auc))
    # plot 0 to 1 line
    ax.plot([0, n], [0, 1], 'k--')
    ax.set_xlabel("N\n(Best proportion: %.3f)" % (_x[best_index] / (n+1)))
    ax.set_ylabel("Explained variance (ratio)\n(cutoff=%.2f)" % cutoff)
    ax.grid()
    ax.legend()


def _annotate_on_magnitude(model, labels, n_samples_annotate, ax):
    if len(labels) != model.components_.shape[0]:
        raise ValueError("number of labels: {} passed does not match component PCA shape: {}".format(len(labels),
                                                                                                     model.components_.shape[                                                                                   0]))
    mag = np.linalg.norm(model.components_[:, :2], axis=1)
    selected = np.argpartition(mag, -n_samples_annotate)[-n_samples_annotate:]
    annotate(model.components_[:, 0], model.components_[:, 1], list(labels), selected, ax=ax, word_shorten=15)


def _best_principle_eigenvectors(model, labels, k=5, p=5):
    """Extracts the topk eigenvectors from p PCs"""
    PC = model.components_
    evr = model.explained_variance_ratio_
    xs = []
    label_set = []
    scores = []

    for i in range(p):
        ind_top = np.argpartition(PC[:, i], k//2)[:k//2]
        ind_bot = np.argpartition(PC[:, i], -(k//2))[-(k//2):]
        label_set.append(
            np.hstack((labels[ind_top], labels[ind_bot])))
        scores.append(
            np.hstack((PC[ind_top, i], PC[ind_bot, i])))
        xs.append(["PC{}\n({:0.1f}%)".format(i+1, evr[i]*100)] * k)

    return join(*xs), join(*scores), join(*label_set)


def _best_eigenvector_plot(x, y, labels, ax, nk=(6, 5)):
    n_samples, n_pcs = nk

    ax.scatter(x, y)
    ax.hlines(0, -.5, n_pcs - .5, linestyle="--")
    annotate(x, y, labels, ax=ax, word_shorten=15)
    ax.set_ylabel("Eigenvector")
    ax.grid()


def overview_pca(model,
                 distance_color=True,
                 labels=None,
                 n_samples_annotate=6,
                 n_pcs=5,
                 ax_size=4):
    """Provides an overview plot from a PCA result.

    Parameters
    ----------
    model : sklearn.decomposition.PCA
        A fitted PCA model.
    distance_color : bool, default=True
        If True, plots the magnitude of each PC as a color
    labels : np.ndarray (n,) of str / pd.Series / list / tuple, optional
        If not None, provides a label for every PC component (dimension), and annotates
        the most 'outlier' like samples in plot 1
    n_samples_annotate : int, default=10
        Defines the number of labels to show if `labels` is not None in plot 1
    n_pcs : int, default=5
        The number of principle components to consider in plot 3
    ax_size : int, default=4
        The default size for each axes.

    Other Parameters
    ----------------
    scatter_kws : dict
        keywords to pass to `plt.scatter`
    """
    instance_check((n_samples_annotate, n_pcs, ax_size), int)
    instance_check(distance_color, bool)
    instance_check(labels, (type(None), np.ndarray, pd.Series, list, tuple))
    nonnegative(n_samples_annotate)
    nonnegative(n_pcs)
    nonnegative(ax_size)

    if labels is not None:
        fig, axes = gridplot(3, ax_size=ax_size)
    else:
        fig, axes = gridplot(2, ax_size=ax_size)

    if n_samples_annotate > model.n_components_:
        n_samples_annotate = model.n_components_
    if n_pcs > model.n_components_:
        n_pcs = model.n_components_

    # 1 plot the scatter of PC
    _plot_pca_scatter(model, axes[0], distance_color)
    # 2 plot the line AUC for explained variance
    _explained_variance_plot(model, axes[1])
    # if annotate, we annotate the scatter plot with samples.
    if labels is not None:
        # check to make sure labels is same length as components
        _annotate_on_magnitude(model, labels, n_samples_annotate, axes[0])
        # 3 plot the top N components by the `most important eigenvector values`
        _x3, _y3, _sel_labels = _best_principle_eigenvectors(model, labels=labels, k=n_samples_annotate, p=n_pcs)
        _best_eigenvector_plot(_x3, _y3, _sel_labels, axes[-1], nk=(n_samples_annotate, n_pcs))
        axes[-1].set_title("Top {} eigenvectors".format(n_samples_annotate))

    fig.tight_layout()
