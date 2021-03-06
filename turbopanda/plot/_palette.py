#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles matplotlib colors and generates useful palettes."""
import itertools as it
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors

from turbopanda.utils import instance_check, unique_ordered

__all__ = ("color_qualitative", "cat_array_to_color", "palette_cmap")


def _tuple_to_hex(t):
    return "#%02x%02x%02x" % (int(t[0]), int(t[1]), int(t[2]))


def _color_scale_off_pair(cmap):
    return _colormap_to_hex(cm.get_cmap(cmap)(np.linspace(0.25, 0.75, 2)))


def _luminance(arr):
    return np.dot(arr, np.array([0.299, 0.587, 0.114, 0.0]))


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
        return [
            "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in cm_array
        ]
    elif isinstance(cm_array, pd.Series):
        # convert to ndarray
        cm_array = pd.DataFrame(np.vstack(cm_array.values))
    if isinstance(cm_array, pd.DataFrame):
        if cm_array.shape[1] == 4:
            return [
                "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
                for idx, (r, g, b, _) in cm_array.iterrows()
            ]
        elif cm_array.shape[1] == 3:
            return [
                "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
                for idx, (r, g, b) in cm_array.iterrows()
            ]
        else:
            raise ValueError(
                "dimension of 'cm_array' must be 3 or 4, not {}".format(
                    cm_array.shape[1]
                )
            )


def lighten(c, frac_change=0.3):
    """Given a color name, returns a slightly lighter version of that color.

    c can be a str or list of str.
    """
    x = np.asarray(colors.to_rgba(c))
    other = np.array([frac_change, frac_change, frac_change, 0])
    clipped = np.clip(x + other, 0.0, 1.0)
    # convert to hex str and return
    return colors.rgb2hex(clipped)


def darken(c, frac_change=0.3):
    """Given a color name, returns a slightly lighter version of that color"""
    x = np.asarray(colors.to_rgba(c))
    other = np.array([frac_change, frac_change, frac_change, 0])
    clipped = np.clip(x - other, 0.0, 1.0)
    # convert to hex str and return
    return colors.rgb2hex(clipped)


def autoshade(c, frac_change=0.3):
    """Given a color name, returns a slightly lighter OR darker version of that color"""
    x = np.asarray(colors.to_rgba(c))
    # determine luminousity
    lum = _luminance(x)
    other = np.array([frac_change, frac_change, frac_change, 0])
    if lum > 0.5:
        clipped = np.clip(x - other, 0.0, 1.0)
    else:
        clipped = np.clip(x + other, 0.0, 1.0)
    return colors.rgb2hex(clipped)


def noncontrast(c):
    """Given colour c, find best noncontrasting colour (white or black).

    References
    ----------
    https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
    """
    if isinstance(c, str):
        _c = np.asarray(colors.to_rgba(c))
    else:
        _c = np.asarray(c)

    # calculate perpective luminance
    lum_weights = np.array([0.299, 0.587, 0.114, 0.0])
    luminance = np.dot(_c, lum_weights)
    # if luminance is high, use black font, else use white font
    if luminance < 0.5:
        return "#000000"
    else:
        return "#ffffff"


def contrast(c):
    """Given colour c, find best contrasting colour (white or black).

    References
    ----------
    https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
    """
    if isinstance(c, str):
        _c = np.asarray(colors.to_rgba(c))
    else:
        _c = np.asarray(c)

    # calculate perpective luminance
    luminance = _luminance(_c)
    # if luminance is high, use black font, else use white font
    if luminance > 0.5:
        return "#000000"
    else:
        return "#ffffff"


""" Qualitative methods """


def palette_black(n: int):
    """Returns a qualitiative set of black-white colors"""
    return palette_cmap(n, "Greys")


def palette_red(n: int):
    """Returns a qualitiative set of red colors"""
    return palette_cmap(n, "Reds")


def palette_green(n: int):
    """Returns a qualitiative set of green colors"""
    return palette_cmap(n, "Greens")


def palette_blue(n: int):
    """Returns a qualitiative set of blue colors"""
    return palette_cmap(n, "Blues")


def palette_pairs(n: int):
    """Returns a palette-pair (2 colors), as (darker, lighter)"""
    options_ = ("Greys", "Blues", "Reds", "Greens", "Purples", "Oranges")
    cols = map(_color_scale_off_pair, options_)
    return list(it.islice(it.cycle(cols), 0, n))


def palette_cmap(n: int, cmap: str, rstart=0., rend=1.):
    """given n, calculate the linspace searched for monocolor scales"""
    return _colormap_to_hex(cm.get_cmap(cmap)(np.linspace(rstart, rend, n)))


def color_qualitative(n: Union[int, List, Tuple], sharp: bool = True) -> List[str]:
    """Generates a qualitative palette generator as hex.

    Parameters
    ----------
    n : int, list or tuple
        The number of hex colors to return, or the list/tuple of elements.
    sharp : bool
        If True, only uses strong/sharp colors, else uses pastelly colors.

    Returns
    -------
    L : list
        list of hex colors of length (n,).
    """
    instance_check(n, (int, list, tuple))
    instance_check(sharp, bool)

    if isinstance(n, (list, tuple)):
        n = len(n)

    lt8_sharp = ("Accent", "Dark2")
    lt8_pastel = ("Pastel2", "Set2")
    # lt9 = ('Set1', 'Pastel1')
    # lt10 = ['tab10']
    # lt12 = ['Set3']
    lt20 = ("tab20", "tab20b", "tab20c")
    # choose random cmap
    if n <= 8 and sharp:
        return _colormap_to_hex(
            getattr(cm, np.random.choice(lt8_sharp))(np.linspace(0, 1, n))
        )
    elif n <= 8 and not sharp:
        return _colormap_to_hex(
            getattr(cm, np.random.choice(lt8_pastel))(np.linspace(0, 1, n))
        )
    elif n <= 9 and sharp:
        return palette_cmap(n, "Set1")
    elif n <= 9 and not sharp:
        return palette_cmap(n, "Pastel1")
    elif n <= 10:
        return palette_cmap(n, "tab10")
    elif n <= 12:
        return palette_cmap(n, "Set3")
    elif n <= 20:
        return _colormap_to_hex(
            getattr(cm, np.random.choice(lt20))(np.linspace(0, 1, n))
        )
    else:
        # we cycle one of the lt20s
        return list(
            it.islice(
                it.cycle(
                    _colormap_to_hex(
                        getattr(cm, np.random.choice(lt20))(np.linspace(0, 1, 20))
                    )
                ),
                0,
                n,
            )
        )


def cat_array_to_color(array, cmap="Blues", rstart=0., rend=1.):
    """Given some list/array of values, find some way of mapping this to colour values

    Parameters
    ----------
    array : np.ndarray
        An array of values.
    cmap : str, dict, list-like
        Either a string or an array referencing the 'unique' colours
        If dict (k = name, v = hex color)
    rstart : float [0..1]
        The start of the color range
    rend : float [0..1]
        The end of the color range
    """
    instance_check(cmap, (str, dict, list, tuple))

    # map to numpy
    _array = (
        np.asarray(array).flatten()
        if not isinstance(array, (np.ndarray, pd.Series))
        else array
    )
    # if boolean, cast as a 'string'
    if _array.dtype.kind == "b":
        _array = _array.astype(np.str)
    if (_array.dtype.kind == "U") | (_array.dtype.kind == "O"):
        # i.e we have a string array
        name_uniq = unique_ordered(_array)
        col_uniq = _colormap_to_hex(cm.get_cmap(cmap)(np.linspace(rstart, rend, len(name_uniq))))
        name_col_map = dict(zip(name_uniq, col_uniq))
        cl = list(map(lambda s: s.replace(s, name_col_map[s]), _array))
        return np.asarray(cl), "discrete"
    else:
        return _array, "continuous"
