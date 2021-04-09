#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for intelligent heatmaps in primitive matplotlib.

Acknowledgements to the matplotlib library:
https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Rectangle
from pandas import factorize


def _hinton_square_bias(x, b=1.):
    b = np.clip(b, 1, 10)
    return x**(1./b)


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Taken directly from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def hinton(C,
           pad=0.01,
           min_square_size=0.01,
           square_bias=1.,
           cluster=False,
           draw_cluster_boxes=False,
           ax=None,
           colorbar=True,
           cmap=mpl.cm.seismic):
    """Plots a Hinton correlation matrix with adjusted square sizes.

    Note that with square bias, y=r**(1/b) where b is the bias, allows for augmentation of
    smaller r values, depending on need.

    Parameters
    ----------
    C : pd.DataFrame(p, p)
        The symmetric (p,p) correlation matrix in the range [-1, 1], with diagonal elements 1
    pad : float, default (0.01)
        The padding between each square
    min_square_size : float, default (0.01)
        The minimize size of each square
    square_bias : float, default (1)
        Bias parameter to augment square size, default has no effect
    cluster : bool, default=False
        Whether to use hierarchical clustering on the data
    draw_cluster_boxes : bool, default=False
        If `cluster`, also draws rectangle boxes around hierarchical groups
    ax : matplotlib.ax.Axes, default None
        A plot to draw upon
    colorbar : bool, default=True
        Whether to draw a colorbar if an ax isnt provided
    cmap : Colormap, default 'seismic'
        Which matplotlib color map to use

    Returns
    -------
    fig : matplotlib.Figure
        The main figure object
    ax : matplotlib axes
        The ax created if no ax was provided
    """

    # check that C is symmetric
    if C.shape[0] != C.shape[1]:
        raise ValueError("C dimensions {} do not match".format(C.shape))

    no_plot = ax is None

    if no_plot:
        fig, ax = plt.subplots(figsize=(12, 10))

    # if clustering, sort the data first
    if cluster:
        Z = linkage(C)
        dn = dendrogram(Z, no_plot=True, get_leaves=True)
        # use the leaves to sort the data matrix
        data = np.asarray(C.iloc[dn['leaves'], dn['leaves']])
        columns = C.columns[dn['leaves']]
    else:
        data = np.asarray(C)
        columns = C.columns

    p = data.shape[0]
    box_size = 1./p
    tick_pos = pad/2. + (np.arange(p)/p) + (box_size-pad)/2.
    max_size = box_size - pad

    # iterate through the grid and draw rectangles
    for i in range(p):
        for j in range(p):
            # convert correlation from [-1, 1] to [0, 1] range
            r_01 = (data[i, j] / 2.) + .5
            # calculate appropriate size
            size = np.clip(
                max_size * _hinton_square_bias(np.abs(data[i, j]), square_bias),
                min_square_size,
                max_size
            )
            x = tick_pos[i] - (size/2.)
            y = tick_pos[p-j-1] - (size/2)
            # add rect patch
            ax.add_patch(Rectangle((x, y), size, size, color=cmap(r_01)))

    # draw line rectangles around cluster groups if we have clustered
    if cluster and draw_cluster_boxes:
        _F, dX = factorize(dn['leaves_color_list'])
        # compute the indices of where the rectangles should start
        idx_nonz = np.hstack((np.array([0], dtype=int), np.nonzero(np.diff(_F))[0] + 1))
        # x,y points
        pX = np.linspace(0, 1., p + 1)
        # calculate size of the boxes
        _sf = np.diff(idx_nonz)
        _boxes = np.hstack((_sf, np.array([p - np.sum(_sf)])))
        _actual_sizes = _boxes / p

        # iterate through and draw boxes.
        for i in range(len(idx_nonz)):
            x = pX[idx_nonz][i]
            y = 1 - x - box_size * _boxes[i]
            ax.add_patch(Rectangle((x, y), _actual_sizes[i], _actual_sizes[i],
                                   facecolor='none', edgecolor='k',
                                   linestyle='dashed', linewidth=2))

    # modify ticks and labels
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(columns[::-1])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    if no_plot and colorbar:
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1., vmax=1.),
                                  cmap=cmap),
            ax=ax, orientation='vertical', shrink=.75)
        cb.set_label(r"$r$", fontsize=15)

    if no_plot:
        return fig, ax
