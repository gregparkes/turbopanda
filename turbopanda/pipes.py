#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:17:26 2019

@author: gparkes
"""

from sklearn import preprocessing


__all__ = ["ml_pipe"]


def ml_pipe(mp, X_s, y_s, preprocessor="scale"):
    """
    Creates a 'delay' pipe that will make your data 'machine-learning-ready'.

    Parameters
    --------
    mp : turb.MetaPanda
        The MetaPanda object
    X_s : list of str, pd.Index
        A list of x-features
    y_s : str/list of str, pd.Index
        A list of y-feature(s)
    preprocessor : str
        Name of preprocessing: default 'scale', choose from
            [power_transform, minmax_scale, maxabs_scale, robust_scale,
             quantile_transform, scale, normalize]
            Choose from sklearn.preprocessing.
    """
    # fetch columns
    y_f = mp.view(y_s)
    if hasattr(preprocessing, preprocessor):
        # try to get function
        preproc_f = getattr(preprocessing, preprocessor.lower())
    else:
        raise ValueError("preprocessor function '{}' not found in sklearn.preprocessing".format(preprocessor))

    return [
        ("drop", (object, "_id$", "_ID$"), {}),
        ("apply", ("dropna",), {"subset": y_f}),
        ("transform", (lambda x: x.fillna(x.mean()), X_s), {}),
        ("transform", (preproc_f,), {"selector": X_s, "whole": True}),
        ("transform", (lambda y: y - y.mean(), y_s), {}),
    ]
