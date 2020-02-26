#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles matplotlib colors and generates useful palettes."""

import itertools as it
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from typing import List


def _colormap_to_hex(cm_array: np.ndarray):
    """
    Given a colormap array arranged as:
        [[r, g, b, a],
         [r, g, b, a],
         ............,
         [r, g, b, a]]

    Computes the hexadecimal for each row and returns as list
    """
    if isinstance(cm_array, np.ndarray):
        return ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in cm_array]
    elif isinstance(cm_array, pd.Series):
        # convert to ndarray
        cm_array = pd.DataFrame(np.vstack(cm_array.values))
    if isinstance(cm_array, pd.DataFrame):
        if cm_array.shape[1] == 4:
            return ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for idx, (r, g, b, _) in
                    cm_array.iterrows()]
        elif cm_array.shape[1] == 3:
            return ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for idx, (r, g, b) in
                    cm_array.iterrows()]
        else:
            raise ValueError("dimension of 'cm_array' must be 3 or 4, not {}".format(cm_array.shape[1]))


def color_qualitative(n: int, sharp: bool = True) -> List:
    """Generates a qualitative palette generator as hex.

    Parameters
    ----------
    n : int
        The number of hex colors to return
    sharp : bool
        If True, only uses strong/sharp colors, else uses pastelly colors.

    Returns
    -------
    L : list
        list of hex colors of length (n,).
    """
    lt8_sharp = ('Accent', 'Dark2')
    lt8_pastel = ('Pastel2', 'Set2')
    # lt9 = ('Set1', 'Pastel1')
    # lt10 = ['tab10']
    # lt12 = ['Set3']
    lt20 = ('tab20', 'tab20b', 'tab20c')
    # choose random cmap
    if n <= 8 and sharp:
        return _colormap_to_hex(getattr(cm, np.random.choice(lt8_sharp))(np.linspace(0, 1, n)))
    elif n <= 8 and not sharp:
        return _colormap_to_hex(getattr(cm, np.random.choice(lt8_pastel))(np.linspace(0, 1, n)))
    elif n <= 9 and sharp:
        return _colormap_to_hex(cm.Set1(np.linspace(0, 1, n)))
    elif n <= 9 and not sharp:
        return _colormap_to_hex(cm.Pastel1(np.linspace(0, 1, n)))
    elif n <= 10:
        return _colormap_to_hex(cm.tab10(np.linspace(0, 1, n)))
    elif n <= 12:
        return _colormap_to_hex(cm.Set3(np.linspace(0, 1, n)))
    elif n <= 20:
        return _colormap_to_hex(getattr(cm, np.random.choice(lt20))(np.linspace(0, 1, n)))
    else:
        # we cycle one of the lt20s
        return list(it.islice(it.cycle(mf.colormap_to_hex(getattr(cm, np.random.choice(lt20))(np.linspace(0, 1, 20)))), 0, n))
