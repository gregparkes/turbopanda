#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saves figures with a matplotlib.Figure handle."""

import os
import warnings
from typing import Tuple

import matplotlib.pyplot as plt

from turbopanda.utils import belongs, instance_check


def save(fig_obj: plt.Figure,
         plot_type: str,
         name: str = "example1",
         save_types: Tuple[str, ...] = ("png", "pdf"),
         fp: str = "./",
         dpi: int = 360,
         savemode: str = "first") -> bool:
    """Saves a matplotlib figure in many formats.

    Given a matplotlib.Figure object, save appropriate numbers of Figures to the respective
    folders.

    Parameters
    ----------
    fig_obj : plt.Figure
        The figure object to save.
    plot_type : str
        Choose from:
            {"scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
            "box", "line", "venn", "multi", "pie"}
    name : str, optional
        The name of the file, this may be added to based on the other parameters
    save_types : tuple of str, optional
        Choose any from {"png", "pdf", "svg", "eps", "ps"}
    fp : str, optional
        The file path to the root directory of saving images
    dpi : int, optional
        The resolution in dots per inch; set to high if you want a good image
    savemode : str, optional
        Choose from {'first', 'update'}
            if first, only saves if file isn't present
            if update, overrides saved figure if present

    Warnings
    --------
    UserWarning
        If figure file itself already exists

    Raises
    ------
    IOError
        If the filepath does not exist
    TypeError
        If the arguments do not match their declared type
    ValueError
        If `plot_type`, `savemode` does not belong to an acceptable argument

    Returns
    -------
    success : bool
        Whether it was successful or not
    """

    instance_check(fig_obj, plt.axes.Figure)
    instance_check((plot_type, name, fp, savemode), str)
    instance_check(dpi, int)

    accepted_types = (
        "scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
        "box", "line", "venn", "multi", "pie"
    )
    file_types_supported = ("png", "pdf", "svg", "eps", "ps")
    accepted_savemodes = ('first', 'update')

    instance_check(fig_obj, plt.Figure)
    instance_check(name, str)
    instance_check(fp, str)
    belongs(plot_type, accepted_types)
    belongs(savemode, accepted_savemodes)

    for st in save_types:
        if st not in file_types_supported:
            TypeError("save_type: [%s] not supported" % st)

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
