#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import os
import warnings
# locals
from .utils import instance_check
import matplotlib.pyplot as plt


__all__ = ['save_figure']


def save_figure(fig_obj,
                plot_type,
                name="example1",
                save_types=("png", "pdf"),
                fp="./",
                dpi=360,
                savemode="first"):
    """
    Given a matplotlib.Figure object, save appropriate numbers of Figures to the respective
    folders.

    Parameters
    -------
    fig : matplotlib.Figure
        The figure object to save.
    plot_type : str
        The type of plot this is, accepted inputs are:
        ["scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
        "box", "line", "venn", "multi", "pie"]
    name : str (optional)
        The name of the file, this may be added to based on the other parameters
    save_types : list (optional)
        Contains every unique save type to use e.g ["png", "pdf", "svg"]..
    fp : str (optional)
        The file path to the root directory of saving images
    dpi : int
        The resolution in dots per inch; set to high if you want a good image
    savemode : str
        ['first', 'update']: if first, only saves if file isn't present, if update,
        overrides saved figure

    Returns
    -------
    success : bool
        Whether it was successful or not
    """
    instance_check(fig_obj, plt.Figure)
    accepted_types = [
        "scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
        "box", "line", "venn", "multi", "pie"
    ]
    file_types_supported = ["png", "pdf", "svg", "eps", "ps"]
    accepted_savemodes = ['first', 'update']

    if plot_type not in accepted_types:
        raise TypeError("plot_type: [%s] not found in accepted types!" % plot_type)

    for st in save_types:
        if st not in file_types_supported:
            TypeError("save_type: [%s] not supported" % st)
    if savemode not in accepted_savemodes:
        raise ValueError("savemode: '{}' not found in {}".format(savemode, accepted_savemodes))

    # correct to ensure filepath has / at end
    if not fp.endswith("/"):
        fp += "/"

    # check whether the filepath exists
    if os.path.exists(fp):
        for t in save_types:
            # if the directory does not exist, create it!
            if not os.path.isdir(fp + "_" + t):
                os.mkdir(fp + "_" + t)
            # check if the figures themselves already exist.
            filename = "{}_{}/{}_{}.{}".format(fp, t, plot_type, name, t)
            if os.path.isfile(filename):
                warnings.warn("Figure: '{}' already exists: Using savemode: {}".format(filename, savemode), UserWarning)
                if savemode == 'update':
                    fig_obj.savefig(filename, format=t, bbox_inches='tight', dpi=dpi)
            else:
                # make the file
                fig_obj.savefig(filename, format=t, bbox_inches="tight", dpi=dpi)
    else:
        raise IOError("filepath: [%s] does not exist." % fp)
    return True
