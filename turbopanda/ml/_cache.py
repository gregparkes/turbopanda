#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Caches machine learning models."""


import json
from typing import Dict


def jsonify_validate(validate: Dict):
    """JSONify the results from sklearn.cross_validate.

    Assumes estimators and training score are returned.

    """
    # columns

    #jsonify the estimator

    return json.dumps(validate)


