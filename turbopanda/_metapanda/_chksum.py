#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the checksum operations in Metapanda."""

import numpy as np
import pandas as pd
import hashlib
import warnings


def _seeded_checksum(df):
    """Generates a seeded checksum for a MetaPanda, given it's `df_`.

    TODO Implement into `write_json` and `from_json` the seeded checksum

    Includes columns and some data inside, but not all for computational reasons.
    """
    if df.shape[0] < 10 or df.shape[1] < 3:
        warnings.warn("_seeded_checksum not viable, dataset too small.", UserWarning)
        return '0'
    # set the seed
    np.random.seed(123456789)
    # determine selected columns
    _sel_cols = np.random.choice(df.columns, size=(3,), replace=False)
    # stringify columns
    _str_cols = json.dumps(list(set(df.columns)))
    # set a new seed and determine selected subset of data for seed
    np.random.seed(987654321)
    _sel_data = json.dumps(df.sample(n=10).loc[:, _sel_cols].round(2).to_dict())
    # generate sha256 and add together
    chk1 = hashlib.sha256(_str_cols.encode()).hexdigest()
    chk2 = hashlib.sha256(_sel_data.encode()).hexdigest()
    return chk1 + chk2
