#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles some mutual information calculations."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from turbopanda._metapanda import MetaPanda
from turbopanda.utils import intersect, instance_check


__all__ = ['mutual_info']


def mutual_info(data, x, y, z=None, **kws):
    """Calculates the mutual information between X, Y, and potentially Z.

    Parameters
    ---------
    data : pd.DataFrame / MetaPanda
        The full dataset.
    x : (str, list, tuple, pd.Index), optional
        Subset of input(s) for column names.
            if None, uses the full dataset. Y must be None in this case also.
    y : (str, list, tuple, pd.Index)
        Subset of output(s) for column names.
            if None, uses the full dataset (from optional `x` subset)
    z : (str, list, tuple, pd.Index), optional
        set of covariate(s). Covariates are needed to compute conditional mutual information.
            If None, uses standard MI.
    kws : dict
        Additional keywords to pass to drv.information_mutual[_conditional]

    Returns
    -------
    MI : np.ndarray
        mutual information matrix, (same dimensions as |x|, |y|, |z| input).
    """
    from pyitlib import discrete_random_variable as drv

    instance_check(data, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, (str, list, tuple, pd.Index))
    instance_check(z, (type(None), str, list, tuple, pd.Index))

    # downcast if list/tuple/pd.index is of length 1
    x = x[0] if (isinstance(x, (tuple, list, pd.Index)) and len(x) == 1) else x
    y = y[0] if (isinstance(y, (tuple, list, pd.Index)) and len(y) == 1) else y

    # cleaned set with no object or id columns
    _clean = data.view_not("object", '_id$', '_ID$', "^counter")
    # assume x, y, z are selectors, perform intersection between cleaned set and the columns we want
    _X = intersect(data.view(x), _clean)
    _Y = intersect(data.view(y), _clean)
    if z is not None:
        _Z = intersect(data.view(z), _clean)
        # calculate conditional mutual information
        _mi = drv.information_mutual_conditional(
            data[_X].T, data[_Y].T, data[_Z].T, cartesian_product=True, **kws
        )
    else:
        _mi = drv.information_mutual(
            data[_X].T, data[_Y].T, cartesian_product=True, **kws
        )

    # depending on the dimensions of x, y and z, can be up to 3D
    return _mi
